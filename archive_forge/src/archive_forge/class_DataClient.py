import math
import logging
import queue
import threading
import warnings
import grpc
from collections import OrderedDict
from typing import Any, Callable, Dict, TYPE_CHECKING, Optional, Union
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.debug import log_once
class DataClient:

    def __init__(self, client_worker: 'Worker', client_id: str, metadata: list):
        """Initializes a thread-safe datapath over a Ray Client gRPC channel.

        Args:
            client_worker: The Ray Client worker that manages this client
            client_id: the generated ID representing this client
            metadata: metadata to pass to gRPC requests
        """
        self.client_worker = client_worker
        self._client_id = client_id
        self._metadata = metadata
        self.data_thread = self._start_datathread()
        self.outstanding_requests: Dict[int, Any] = OrderedDict()
        self.lock = threading.Lock()
        self.cv = threading.Condition(lock=self.lock)
        self.request_queue = self._create_queue()
        self.ready_data: Dict[int, Any] = {}
        self.asyncio_waiting_data: Dict[int, ResponseCallable] = {}
        self._in_shutdown = False
        self._req_id = 0
        self._last_exception = None
        self._acknowledge_counter = 0
        self.data_thread.start()

    def _next_id(self) -> int:
        assert self.lock.locked()
        self._req_id += 1
        if self._req_id > INT32_MAX:
            self._req_id = 1
        assert self._req_id != 0
        return self._req_id

    def _start_datathread(self) -> threading.Thread:
        return threading.Thread(target=self._data_main, name='ray_client_streaming_rpc', args=(), daemon=True)

    def _requests(self):
        while True:
            req = self.request_queue.get()
            if req is None:
                return
            req_type = req.WhichOneof('type')
            if req_type == 'put':
                yield from chunk_put(req)
            elif req_type == 'task':
                yield from chunk_task(req)
            else:
                yield req

    def _data_main(self) -> None:
        reconnecting = False
        try:
            while not self.client_worker._in_shutdown:
                stub = ray_client_pb2_grpc.RayletDataStreamerStub(self.client_worker.channel)
                metadata = self._metadata + [('reconnecting', str(reconnecting))]
                resp_stream = stub.Datapath(self._requests(), metadata=metadata, wait_for_ready=True)
                try:
                    for response in resp_stream:
                        self._process_response(response)
                    return
                except grpc.RpcError as e:
                    reconnecting = self._can_reconnect(e)
                    if not reconnecting:
                        self._last_exception = e
                        return
                    self._reconnect_channel()
        except Exception as e:
            self._last_exception = e
        finally:
            logger.debug('Shutting down data channel.')
            self._shutdown()

    def _process_response(self, response: Any) -> None:
        """
        Process responses from the data servicer.
        """
        if response.req_id == 0:
            logger.debug(f'Got unawaited response {response}')
            return
        if response.req_id in self.asyncio_waiting_data:
            can_remove = True
            try:
                callback = self.asyncio_waiting_data[response.req_id]
                if isinstance(callback, ChunkCollector):
                    can_remove = callback(response)
                elif callback:
                    callback(response)
                if can_remove:
                    del self.asyncio_waiting_data[response.req_id]
            except Exception:
                logger.exception('Callback error:')
            with self.lock:
                if response.req_id in self.outstanding_requests and can_remove:
                    del self.outstanding_requests[response.req_id]
                    self._acknowledge(response.req_id)
        else:
            with self.lock:
                self.ready_data[response.req_id] = response
                self.cv.notify_all()

    def _can_reconnect(self, e: grpc.RpcError) -> bool:
        """
        Processes RPC errors that occur while reading from data stream.
        Returns True if the error can be recovered from, False otherwise.
        """
        if not self.client_worker._can_reconnect(e):
            logger.error('Unrecoverable error in data channel.')
            logger.debug(e)
            return False
        logger.debug('Recoverable error in data channel.')
        logger.debug(e)
        return True

    def _shutdown(self) -> None:
        """
        Shutdown the data channel
        """
        with self.lock:
            self._in_shutdown = True
            self.cv.notify_all()
            callbacks = self.asyncio_waiting_data.values()
            self.asyncio_waiting_data = {}
        if self._last_exception:
            err = ConnectionError(f'Failed during this or a previous request. Exception that broke the connection: {self._last_exception}')
        else:
            err = ConnectionError('Request cannot be fulfilled because the data client has disconnected.')
        for callback in callbacks:
            if callback:
                callback(err)

    def _acknowledge(self, req_id: int) -> None:
        """
        Puts an acknowledge request on the request queue periodically.
        Lock should be held before calling this. Used when an async or
        blocking response is received.
        """
        if not self.client_worker._reconnect_enabled:
            return
        assert self.lock.locked()
        self._acknowledge_counter += 1
        if self._acknowledge_counter % ACKNOWLEDGE_BATCH_SIZE == 0:
            self.request_queue.put(ray_client_pb2.DataRequest(acknowledge=ray_client_pb2.AcknowledgeRequest(req_id=req_id)))

    def _reconnect_channel(self) -> None:
        """
        Attempts to reconnect the gRPC channel and resend outstanding
        requests. First, the server is pinged to see if the current channel
        still works. If the ping fails, then the current channel is closed
        and replaced with a new one.

        Once a working channel is available, a new request queue is made
        and filled with any outstanding requests to be resent to the server.
        """
        try:
            ping_succeeded = self.client_worker.ping_server(timeout=5)
        except grpc.RpcError:
            ping_succeeded = False
        if not ping_succeeded:
            logger.warning('Encountered connection issues in the data channel. Attempting to reconnect.')
            try:
                self.client_worker._connect_channel(reconnecting=True)
            except ConnectionError:
                logger.warning('Failed to reconnect the data channel')
                raise
            logger.debug('Reconnection succeeded!')
        with self.lock:
            self.request_queue = self._create_queue()
            for request in self.outstanding_requests.values():
                self.request_queue.put(request)

    @staticmethod
    def _create_queue():
        return queue.SimpleQueue()

    def close(self) -> None:
        thread = None
        with self.lock:
            self._in_shutdown = True
            self.cv.notify_all()
            if self.request_queue is not None:
                cleanup_request = ray_client_pb2.DataRequest(connection_cleanup=ray_client_pb2.ConnectionCleanupRequest())
                self.request_queue.put(cleanup_request)
                self.request_queue.put(None)
            if self.data_thread is not None:
                thread = self.data_thread
        if thread is not None:
            thread.join()

    def _blocking_send(self, req: ray_client_pb2.DataRequest) -> ray_client_pb2.DataResponse:
        with self.lock:
            self._check_shutdown()
            req_id = self._next_id()
            req.req_id = req_id
            self.request_queue.put(req)
            self.outstanding_requests[req_id] = req
            self.cv.wait_for(lambda: req_id in self.ready_data or self._in_shutdown)
            self._check_shutdown()
            data = self.ready_data[req_id]
            del self.ready_data[req_id]
            del self.outstanding_requests[req_id]
            self._acknowledge(req_id)
        return data

    def _async_send(self, req: ray_client_pb2.DataRequest, callback: Optional[ResponseCallable]=None) -> None:
        with self.lock:
            self._check_shutdown()
            req_id = self._next_id()
            req.req_id = req_id
            self.asyncio_waiting_data[req_id] = callback
            self.outstanding_requests[req_id] = req
            self.request_queue.put(req)

    def _check_shutdown(self):
        assert self.lock.locked()
        if not self._in_shutdown:
            return
        self.lock.release()
        if threading.current_thread().ident == self.data_thread.ident:
            return
        from ray.util import disconnect
        disconnect()
        self.lock.acquire()
        if self._last_exception is not None:
            msg = f"Request can't be sent because the Ray client has already been disconnected due to an error. Last exception: {self._last_exception}"
        else:
            msg = "Request can't be sent because the Ray client has already been disconnected."
        raise ConnectionError(msg)

    def Init(self, request: ray_client_pb2.InitRequest, context=None) -> ray_client_pb2.InitResponse:
        datareq = ray_client_pb2.DataRequest(init=request)
        resp = self._blocking_send(datareq)
        return resp.init

    def PrepRuntimeEnv(self, request: ray_client_pb2.PrepRuntimeEnvRequest, context=None) -> ray_client_pb2.PrepRuntimeEnvResponse:
        datareq = ray_client_pb2.DataRequest(prep_runtime_env=request)
        resp = self._blocking_send(datareq)
        return resp.prep_runtime_env

    def ConnectionInfo(self, context=None) -> ray_client_pb2.ConnectionInfoResponse:
        datareq = ray_client_pb2.DataRequest(connection_info=ray_client_pb2.ConnectionInfoRequest())
        resp = self._blocking_send(datareq)
        return resp.connection_info

    def GetObject(self, request: ray_client_pb2.GetRequest, context=None) -> ray_client_pb2.GetResponse:
        datareq = ray_client_pb2.DataRequest(get=request)
        resp = self._blocking_send(datareq)
        return resp.get

    def RegisterGetCallback(self, request: ray_client_pb2.GetRequest, callback: ResponseCallable) -> None:
        if len(request.ids) != 1:
            raise ValueError(f'RegisterGetCallback() must have exactly 1 Object ID. Actual: {request}')
        datareq = ray_client_pb2.DataRequest(get=request)
        collector = ChunkCollector(callback=callback, request=datareq)
        self._async_send(datareq, collector)

    def PutObject(self, request: ray_client_pb2.PutRequest, context=None) -> ray_client_pb2.PutResponse:
        datareq = ray_client_pb2.DataRequest(put=request)
        resp = self._blocking_send(datareq)
        return resp.put

    def ReleaseObject(self, request: ray_client_pb2.ReleaseRequest, context=None) -> None:
        datareq = ray_client_pb2.DataRequest(release=request)
        self._async_send(datareq)

    def Schedule(self, request: ray_client_pb2.ClientTask, callback: ResponseCallable):
        datareq = ray_client_pb2.DataRequest(task=request)
        self._async_send(datareq, callback)

    def Terminate(self, request: ray_client_pb2.TerminateRequest) -> ray_client_pb2.TerminateResponse:
        req = ray_client_pb2.DataRequest(terminate=request)
        resp = self._blocking_send(req)
        return resp.terminate

    def ListNamedActors(self, request: ray_client_pb2.ClientListNamedActorsRequest) -> ray_client_pb2.ClientListNamedActorsResponse:
        req = ray_client_pb2.DataRequest(list_named_actors=request)
        resp = self._blocking_send(req)
        return resp.list_named_actors