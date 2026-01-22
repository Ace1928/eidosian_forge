import base64
import functools
import gc
import inspect
import json
import logging
import math
import pickle
import queue
import threading
import time
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, Set, Union
import grpc
import ray
import ray._private.state
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray import cloudpickle
from ray._private import ray_constants
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.ray_constants import env_integer
from ray._private.ray_logging import setup_logger
from ray._private.services import canonicalize_bootstrap_address_or_die
from ray._private.tls_utils import add_port_to_grpc_server
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import DataServicer
from ray.util.client.server.logservicer import LogstreamServicer
from ray.util.client.server.proxier import serve_proxier
from ray.util.client.server.server_pickler import dumps_from_server, loads_from_client
from ray.util.client.server.server_stubs import current_server
class RayletServicer(ray_client_pb2_grpc.RayletDriverServicer):

    def __init__(self, ray_connect_handler: Callable):
        """Construct a raylet service

        Args:
           ray_connect_handler: Function to connect to ray cluster
        """
        self.object_refs: Dict[str, Dict[bytes, ray.ObjectRef]] = defaultdict(dict)
        self.client_side_ref_map: Dict[str, Dict[bytes, bytes]] = defaultdict(dict)
        self.function_refs = {}
        self.actor_refs: Dict[bytes, ray.ActorHandle] = {}
        self.actor_owners: Dict[str, Set[bytes]] = defaultdict(set)
        self.registered_actor_classes = {}
        self.named_actors = set()
        self.state_lock = threading.Lock()
        self.ray_connect_handler = ray_connect_handler
        self.response_caches: Dict[str, ResponseCache] = defaultdict(ResponseCache)

    def Init(self, request: ray_client_pb2.InitRequest, context=None) -> ray_client_pb2.InitResponse:
        if request.job_config:
            job_config = pickle.loads(request.job_config)
            job_config._client_job = True
        else:
            job_config = None
        current_job_config = None
        with disable_client_hook():
            if ray.is_initialized():
                worker = ray._private.worker.global_worker
                current_job_config = worker.core_worker.get_job_config()
            else:
                extra_kwargs = json.loads(request.ray_init_kwargs or '{}')
                try:
                    self.ray_connect_handler(job_config, **extra_kwargs)
                except Exception as e:
                    logger.exception('Running Ray Init failed:')
                    return ray_client_pb2.InitResponse(ok=False, msg=f'Call to `ray.init()` on the server failed with: {e}')
        if job_config is None:
            return ray_client_pb2.InitResponse(ok=True)
        job_config = job_config._get_proto_job_config()
        if current_job_config:
            job_uris = set(job_config.runtime_env_info.uris.working_dir_uri)
            job_uris.update(job_config.runtime_env_info.uris.py_modules_uris)
            current_job_uris = set(current_job_config.runtime_env_info.uris.working_dir_uri)
            current_job_uris.update(current_job_config.runtime_env_info.uris.py_modules_uris)
            if job_uris != current_job_uris and len(job_uris) > 0:
                return ray_client_pb2.InitResponse(ok=False, msg=f"Runtime environment doesn't match request one {job_config.runtime_env_info.uris} current one {current_job_config.runtime_env_info.uris}")
        return ray_client_pb2.InitResponse(ok=True)

    @_use_response_cache
    def KVPut(self, request, context=None) -> ray_client_pb2.KVPutResponse:
        try:
            with disable_client_hook():
                already_exists = ray.experimental.internal_kv._internal_kv_put(request.key, request.value, overwrite=request.overwrite, namespace=request.namespace)
        except Exception as e:
            return_exception_in_context(e, context)
            already_exists = False
        return ray_client_pb2.KVPutResponse(already_exists=already_exists)

    def KVGet(self, request, context=None) -> ray_client_pb2.KVGetResponse:
        try:
            with disable_client_hook():
                value = ray.experimental.internal_kv._internal_kv_get(request.key, namespace=request.namespace)
        except Exception as e:
            return_exception_in_context(e, context)
            value = b''
        return ray_client_pb2.KVGetResponse(value=value)

    @_use_response_cache
    def KVDel(self, request, context=None) -> ray_client_pb2.KVDelResponse:
        try:
            with disable_client_hook():
                deleted_num = ray.experimental.internal_kv._internal_kv_del(request.key, del_by_prefix=request.del_by_prefix, namespace=request.namespace)
        except Exception as e:
            return_exception_in_context(e, context)
            deleted_num = 0
        return ray_client_pb2.KVDelResponse(deleted_num=deleted_num)

    def KVList(self, request, context=None) -> ray_client_pb2.KVListResponse:
        try:
            with disable_client_hook():
                keys = ray.experimental.internal_kv._internal_kv_list(request.prefix, namespace=request.namespace)
        except Exception as e:
            return_exception_in_context(e, context)
            keys = []
        return ray_client_pb2.KVListResponse(keys=keys)

    def KVExists(self, request, context=None) -> ray_client_pb2.KVExistsResponse:
        try:
            with disable_client_hook():
                exists = ray.experimental.internal_kv._internal_kv_exists(request.key, namespace=request.namespace)
        except Exception as e:
            return_exception_in_context(e, context)
            exists = False
        return ray_client_pb2.KVExistsResponse(exists=exists)

    def ListNamedActors(self, request, context=None) -> ray_client_pb2.ClientListNamedActorsResponse:
        with disable_client_hook():
            actors = ray.util.list_named_actors(all_namespaces=request.all_namespaces)
        return ray_client_pb2.ClientListNamedActorsResponse(actors_json=json.dumps(actors))

    def ClusterInfo(self, request, context=None) -> ray_client_pb2.ClusterInfoResponse:
        resp = ray_client_pb2.ClusterInfoResponse()
        resp.type = request.type
        if request.type == ray_client_pb2.ClusterInfoType.CLUSTER_RESOURCES:
            with disable_client_hook():
                resources = ray.cluster_resources()
            float_resources = {k: float(v) for k, v in resources.items()}
            resp.resource_table.CopyFrom(ray_client_pb2.ClusterInfoResponse.ResourceTable(table=float_resources))
        elif request.type == ray_client_pb2.ClusterInfoType.AVAILABLE_RESOURCES:
            with disable_client_hook():
                resources = ray.available_resources()
            float_resources = {k: float(v) for k, v in resources.items()}
            resp.resource_table.CopyFrom(ray_client_pb2.ClusterInfoResponse.ResourceTable(table=float_resources))
        elif request.type == ray_client_pb2.ClusterInfoType.RUNTIME_CONTEXT:
            ctx = ray_client_pb2.ClusterInfoResponse.RuntimeContext()
            with disable_client_hook():
                rtc = ray.get_runtime_context()
                ctx.job_id = ray._private.utils.hex_to_binary(rtc.get_job_id())
                ctx.node_id = ray._private.utils.hex_to_binary(rtc.get_node_id())
                ctx.namespace = rtc.namespace
                ctx.capture_client_tasks = rtc.should_capture_child_tasks_in_placement_group
                ctx.gcs_address = rtc.gcs_address
                ctx.runtime_env = rtc.get_runtime_env_string()
            resp.runtime_context.CopyFrom(ctx)
        else:
            with disable_client_hook():
                resp.json = self._return_debug_cluster_info(request, context)
        return resp

    def _return_debug_cluster_info(self, request, context=None) -> str:
        """Handle ClusterInfo requests that only return a json blob."""
        data = None
        if request.type == ray_client_pb2.ClusterInfoType.NODES:
            data = ray.nodes()
        elif request.type == ray_client_pb2.ClusterInfoType.IS_INITIALIZED:
            data = ray.is_initialized()
        elif request.type == ray_client_pb2.ClusterInfoType.TIMELINE:
            data = ray.timeline()
        elif request.type == ray_client_pb2.ClusterInfoType.PING:
            data = {}
        elif request.type == ray_client_pb2.ClusterInfoType.DASHBOARD_URL:
            data = {'dashboard_url': ray._private.worker.get_dashboard_url()}
        else:
            raise TypeError('Unsupported cluster info type')
        return json.dumps(data)

    def release(self, client_id: str, id: bytes) -> bool:
        with self.state_lock:
            if client_id in self.object_refs:
                if id in self.object_refs[client_id]:
                    logger.debug(f'Releasing object {id.hex()} for {client_id}')
                    del self.object_refs[client_id][id]
                    return True
            if client_id in self.actor_owners:
                if id in self.actor_owners[client_id]:
                    logger.debug(f'Releasing actor {id.hex()} for {client_id}')
                    self.actor_owners[client_id].remove(id)
                    if self._can_remove_actor_ref(id):
                        logger.debug(f'Deleting reference to actor {id.hex()}')
                        del self.actor_refs[id]
                    return True
            return False

    def release_all(self, client_id):
        with self.state_lock:
            self._release_objects(client_id)
            self._release_actors(client_id)
        gc.collect()

    def _can_remove_actor_ref(self, actor_id_bytes):
        no_owner = not any((actor_id_bytes in actor_list for actor_list in self.actor_owners.values()))
        return no_owner and actor_id_bytes not in self.named_actors

    def _release_objects(self, client_id):
        if client_id not in self.object_refs:
            logger.debug(f'Releasing client with no references: {client_id}')
            return
        count = len(self.object_refs[client_id])
        del self.object_refs[client_id]
        if client_id in self.client_side_ref_map:
            del self.client_side_ref_map[client_id]
        if client_id in self.response_caches:
            del self.response_caches[client_id]
        logger.debug(f'Released all {count} objects for client {client_id}')

    def _release_actors(self, client_id):
        if client_id not in self.actor_owners:
            logger.debug(f'Releasing client with no actors: {client_id}')
            return
        count = 0
        actors_to_remove = self.actor_owners.pop(client_id)
        for id_bytes in actors_to_remove:
            count += 1
            if self._can_remove_actor_ref(id_bytes):
                logger.debug(f'Deleting reference to actor {id_bytes.hex()}')
                del self.actor_refs[id_bytes]
        logger.debug(f'Released all {count} actors for client: {client_id}')

    @_use_response_cache
    def Terminate(self, req, context=None):
        if req.WhichOneof('terminate_type') == 'task_object':
            try:
                object_ref = self.object_refs[req.client_id][req.task_object.id]
                with disable_client_hook():
                    ray.cancel(object_ref, force=req.task_object.force, recursive=req.task_object.recursive)
            except Exception as e:
                return_exception_in_context(e, context)
        elif req.WhichOneof('terminate_type') == 'actor':
            try:
                actor_ref = self.actor_refs[req.actor.id]
                with disable_client_hook():
                    ray.kill(actor_ref, no_restart=req.actor.no_restart)
            except Exception as e:
                return_exception_in_context(e, context)
        else:
            raise RuntimeError('Client requested termination without providing a valid terminate_type')
        return ray_client_pb2.TerminateResponse(ok=True)

    def _async_get_object(self, request: ray_client_pb2.GetRequest, client_id: str, req_id: int, result_queue: queue.Queue, context=None) -> Optional[ray_client_pb2.GetResponse]:
        """Attempts to schedule a callback to push the GetResponse to the
        main loop when the desired object is ready. If there is some failure
        in scheduling, a GetResponse will be immediately returned.
        """
        if len(request.ids) != 1:
            raise ValueError(f'Async get() must have exactly 1 Object ID. Actual: {request}')
        rid = request.ids[0]
        ref = self.object_refs[client_id].get(rid, None)
        if not ref:
            return ray_client_pb2.GetResponse(valid=False, error=cloudpickle.dumps(ValueError(f'ClientObjectRef with id {rid} not found for client {client_id}')))
        try:
            logger.debug('async get: %s' % ref)
            with disable_client_hook():

                def send_get_response(result: Any) -> None:
                    """Pushes GetResponses to the main DataPath loop to send
                    to the client. This is called when the object is ready
                    on the server side."""
                    try:
                        serialized = dumps_from_server(result, client_id, self)
                        total_size = len(serialized)
                        assert total_size > 0, 'Serialized object cannot be zero bytes'
                        total_chunks = math.ceil(total_size / OBJECT_TRANSFER_CHUNK_SIZE)
                        for chunk_id in range(request.start_chunk_id, total_chunks):
                            start = chunk_id * OBJECT_TRANSFER_CHUNK_SIZE
                            end = min(total_size, (chunk_id + 1) * OBJECT_TRANSFER_CHUNK_SIZE)
                            get_resp = ray_client_pb2.GetResponse(valid=True, data=serialized[start:end], chunk_id=chunk_id, total_chunks=total_chunks, total_size=total_size)
                            chunk_resp = ray_client_pb2.DataResponse(get=get_resp, req_id=req_id)
                            result_queue.put(chunk_resp)
                    except Exception as exc:
                        get_resp = ray_client_pb2.GetResponse(valid=False, error=cloudpickle.dumps(exc))
                        resp = ray_client_pb2.DataResponse(get=get_resp, req_id=req_id)
                        result_queue.put(resp)
                ref._on_completed(send_get_response)
                return None
        except Exception as e:
            return ray_client_pb2.GetResponse(valid=False, error=cloudpickle.dumps(e))

    def GetObject(self, request: ray_client_pb2.GetRequest, context):
        metadata = {k: v for k, v in context.invocation_metadata()}
        client_id = metadata.get('client_id')
        if client_id is None:
            yield ray_client_pb2.GetResponse(valid=False, error=cloudpickle.dumps(ValueError('client_id is not specified in request metadata')))
        else:
            yield from self._get_object(request, client_id)

    def _get_object(self, request: ray_client_pb2.GetRequest, client_id: str):
        objectrefs = []
        for rid in request.ids:
            ref = self.object_refs[client_id].get(rid, None)
            if ref:
                objectrefs.append(ref)
            else:
                yield ray_client_pb2.GetResponse(valid=False, error=cloudpickle.dumps(ValueError(f'ClientObjectRef {rid} is not found for client {client_id}')))
                return
        try:
            logger.debug('get: %s' % objectrefs)
            with disable_client_hook():
                items = ray.get(objectrefs, timeout=request.timeout)
        except Exception as e:
            yield ray_client_pb2.GetResponse(valid=False, error=cloudpickle.dumps(e))
            return
        serialized = dumps_from_server(items, client_id, self)
        total_size = len(serialized)
        assert total_size > 0, 'Serialized object cannot be zero bytes'
        total_chunks = math.ceil(total_size / OBJECT_TRANSFER_CHUNK_SIZE)
        for chunk_id in range(request.start_chunk_id, total_chunks):
            start = chunk_id * OBJECT_TRANSFER_CHUNK_SIZE
            end = min(total_size, (chunk_id + 1) * OBJECT_TRANSFER_CHUNK_SIZE)
            yield ray_client_pb2.GetResponse(valid=True, data=serialized[start:end], chunk_id=chunk_id, total_chunks=total_chunks, total_size=total_size)

    def PutObject(self, request: ray_client_pb2.PutRequest, context=None) -> ray_client_pb2.PutResponse:
        """gRPC entrypoint for unary PutObject"""
        return self._put_object(request.data, request.client_ref_id, '', request.owner_id, context)

    def _put_object(self, data: Union[bytes, bytearray], client_ref_id: bytes, client_id: str, owner_id: bytes, context=None):
        """Put an object in the cluster with ray.put() via gRPC.

        Args:
            data: Pickled data. Can either be bytearray if this is called
              from the dataservicer, or bytes if called from PutObject.
            client_ref_id: The id associated with this object on the client.
            client_id: The client who owns this data, for tracking when to
              delete this reference.
            owner_id: The owner id of the object.
            context: gRPC context.
        """
        try:
            obj = loads_from_client(data, self)
            if owner_id:
                owner = self.actor_refs[owner_id]
            else:
                owner = None
            with disable_client_hook():
                objectref = ray.put(obj, _owner=owner)
        except Exception as e:
            logger.exception('Put failed:')
            return ray_client_pb2.PutResponse(id=b'', valid=False, error=cloudpickle.dumps(e))
        self.object_refs[client_id][objectref.binary()] = objectref
        if len(client_ref_id) > 0:
            self.client_side_ref_map[client_id][client_ref_id] = objectref.binary()
        logger.debug('put: %s' % objectref)
        return ray_client_pb2.PutResponse(id=objectref.binary(), valid=True)

    def WaitObject(self, request, context=None) -> ray_client_pb2.WaitResponse:
        object_refs = []
        for rid in request.object_ids:
            if rid not in self.object_refs[request.client_id]:
                raise Exception('Asking for a ref not associated with this client: %s' % str(rid))
            object_refs.append(self.object_refs[request.client_id][rid])
        num_returns = request.num_returns
        timeout = request.timeout
        try:
            with disable_client_hook():
                ready_object_refs, remaining_object_refs = ray.wait(object_refs, num_returns=num_returns, timeout=timeout if timeout != -1 else None)
        except Exception as e:
            logger.error(f'Exception {e}')
            return ray_client_pb2.WaitResponse(valid=False)
        logger.debug('wait: %s %s' % (str(ready_object_refs), str(remaining_object_refs)))
        ready_object_ids = [ready_object_ref.binary() for ready_object_ref in ready_object_refs]
        remaining_object_ids = [remaining_object_ref.binary() for remaining_object_ref in remaining_object_refs]
        return ray_client_pb2.WaitResponse(valid=True, ready_object_ids=ready_object_ids, remaining_object_ids=remaining_object_ids)

    def Schedule(self, task: ray_client_pb2.ClientTask, arglist: List[Any], kwargs: Dict[str, Any], context=None) -> ray_client_pb2.ClientTaskTicket:
        logger.debug('schedule: %s %s' % (task.name, ray_client_pb2.ClientTask.RemoteExecType.Name(task.type)))
        try:
            with disable_client_hook():
                if task.type == ray_client_pb2.ClientTask.FUNCTION:
                    result = self._schedule_function(task, arglist, kwargs, context)
                elif task.type == ray_client_pb2.ClientTask.ACTOR:
                    result = self._schedule_actor(task, arglist, kwargs, context)
                elif task.type == ray_client_pb2.ClientTask.METHOD:
                    result = self._schedule_method(task, arglist, kwargs, context)
                elif task.type == ray_client_pb2.ClientTask.NAMED_ACTOR:
                    result = self._schedule_named_actor(task, context)
                else:
                    raise NotImplementedError('Unimplemented Schedule task type: %s' % ray_client_pb2.ClientTask.RemoteExecType.Name(task.type))
                result.valid = True
                return result
        except Exception as e:
            logger.debug('Caught schedule exception', exc_info=True)
            return ray_client_pb2.ClientTaskTicket(valid=False, error=cloudpickle.dumps(e))

    def _schedule_method(self, task: ray_client_pb2.ClientTask, arglist: List[Any], kwargs: Dict[str, Any], context=None) -> ray_client_pb2.ClientTaskTicket:
        actor_handle = self.actor_refs.get(task.payload_id)
        if actor_handle is None:
            raise Exception("Can't run an actor the server doesn't have a handle for")
        method = getattr(actor_handle, task.name)
        opts = decode_options(task.options)
        if opts is not None:
            method = method.options(**opts)
        output = method.remote(*arglist, **kwargs)
        ids = self.unify_and_track_outputs(output, task.client_id)
        return ray_client_pb2.ClientTaskTicket(return_ids=ids)

    def _schedule_actor(self, task: ray_client_pb2.ClientTask, arglist: List[Any], kwargs: Dict[str, Any], context=None) -> ray_client_pb2.ClientTaskTicket:
        remote_class = self.lookup_or_register_actor(task.payload_id, task.client_id, decode_options(task.baseline_options))
        opts = decode_options(task.options)
        if opts is not None:
            remote_class = remote_class.options(**opts)
        with current_server(self):
            actor = remote_class.remote(*arglist, **kwargs)
        self.actor_refs[actor._actor_id.binary()] = actor
        self.actor_owners[task.client_id].add(actor._actor_id.binary())
        return ray_client_pb2.ClientTaskTicket(return_ids=[actor._actor_id.binary()])

    def _schedule_function(self, task: ray_client_pb2.ClientTask, arglist: List[Any], kwargs: Dict[str, Any], context=None) -> ray_client_pb2.ClientTaskTicket:
        remote_func = self.lookup_or_register_func(task.payload_id, task.client_id, decode_options(task.baseline_options))
        opts = decode_options(task.options)
        if opts is not None:
            remote_func = remote_func.options(**opts)
        with current_server(self):
            output = remote_func.remote(*arglist, **kwargs)
        ids = self.unify_and_track_outputs(output, task.client_id)
        return ray_client_pb2.ClientTaskTicket(return_ids=ids)

    def _schedule_named_actor(self, task: ray_client_pb2.ClientTask, context=None) -> ray_client_pb2.ClientTaskTicket:
        assert len(task.payload_id) == 0
        actor = ray.get_actor(task.name, task.namespace or None)
        bin_actor_id = actor._actor_id.binary()
        self.actor_refs[bin_actor_id] = actor
        self.actor_owners[task.client_id].add(bin_actor_id)
        self.named_actors.add(bin_actor_id)
        return ray_client_pb2.ClientTaskTicket(return_ids=[actor._actor_id.binary()])

    def lookup_or_register_func(self, id: bytes, client_id: str, options: Optional[Dict]) -> ray.remote_function.RemoteFunction:
        with disable_client_hook():
            if id not in self.function_refs:
                funcref = self.object_refs[client_id][id]
                func = ray.get(funcref)
                if not inspect.isfunction(func):
                    raise Exception("Attempting to register function that isn't a function.")
                if options is None or len(options) == 0:
                    self.function_refs[id] = ray.remote(func)
                else:
                    self.function_refs[id] = ray.remote(**options)(func)
        return self.function_refs[id]

    def lookup_or_register_actor(self, id: bytes, client_id: str, options: Optional[Dict]):
        with disable_client_hook():
            if id not in self.registered_actor_classes:
                actor_class_ref = self.object_refs[client_id][id]
                actor_class = ray.get(actor_class_ref)
                if not inspect.isclass(actor_class):
                    raise Exception("Attempting to schedule actor that isn't a class.")
                if options is None or len(options) == 0:
                    reg_class = ray.remote(actor_class)
                else:
                    reg_class = ray.remote(**options)(actor_class)
                self.registered_actor_classes[id] = reg_class
        return self.registered_actor_classes[id]

    def unify_and_track_outputs(self, output, client_id):
        if output is None:
            outputs = []
        elif isinstance(output, list):
            outputs = output
        else:
            outputs = [output]
        for out in outputs:
            if out.binary() in self.object_refs[client_id]:
                logger.warning(f'Already saw object_ref {out}')
            self.object_refs[client_id][out.binary()] = out
        return [out.binary() for out in outputs]