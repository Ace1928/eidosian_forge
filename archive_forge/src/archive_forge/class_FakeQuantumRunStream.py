from typing import AsyncIterable, AsyncIterator, Awaitable, List, Sequence, Union
import asyncio
import concurrent
from unittest import mock
import duet
import pytest
import google.api_core.exceptions as google_exceptions
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine.stream_manager import (
from cirq_google.cloud import quantum
class FakeQuantumRunStream:
    """A fake Quantum Engine client which supports QuantumRunStream and CancelQuantumJob."""
    _REQUEST_STOPPED = 'REQUEST_STOPPED'

    def __init__(self) -> None:
        self.all_stream_requests: List[quantum.QuantumRunStreamRequest] = []
        self.all_cancel_requests: List[quantum.CancelQuantumJobRequest] = []
        self._executor = AsyncioExecutor.instance()
        self._request_buffer = duet.AsyncCollector[quantum.QuantumRunStreamRequest]()
        self._request_iterator_stopped: duet.AwaitableFuture[None] = duet.AwaitableFuture()
        self._responses_and_exceptions_future: duet.AwaitableFuture[asyncio.Queue[Union[quantum.QuantumRunStreamResponse, BaseException]]] = duet.AwaitableFuture()

    async def quantum_run_stream(self, requests: AsyncIterator[quantum.QuantumRunStreamRequest], **kwargs) -> Awaitable[AsyncIterable[quantum.QuantumRunStreamResponse]]:
        """Fakes the QuantumRunStream RPC.

        Once a request is received, it is appended to `all_stream_requests`, and the test calling
        `wait_for_requests()` is notified.

        The response is sent when a test calls `reply()` with a `QuantumRunStreamResponse`. If a
        test calls `reply()` with an exception, it is raised here to the `quantum_run_stream()`
        caller.

        This is called from the asyncio thread.
        """
        responses_and_exceptions: asyncio.Queue[Union[quantum.QuantumRunStreamResponse, BaseException]] = asyncio.Queue()
        self._responses_and_exceptions_future.try_set_result(responses_and_exceptions)

        async def read_requests():
            async for request in requests:
                self.all_stream_requests.append(request)
                self._request_buffer.add(request)
            await responses_and_exceptions.put(FakeQuantumRunStream._REQUEST_STOPPED)
            self._request_iterator_stopped.try_set_result(None)

        async def response_iterator():
            asyncio.create_task(read_requests())
            while (message := (await responses_and_exceptions.get())) != FakeQuantumRunStream._REQUEST_STOPPED:
                if isinstance(message, quantum.QuantumRunStreamResponse):
                    yield message
                else:
                    self._responses_and_exceptions_future = duet.AwaitableFuture()
                    raise message
        return response_iterator()

    async def cancel_quantum_job(self, request: quantum.CancelQuantumJobRequest) -> None:
        """Records the cancellation in `cancel_requests`.

        This is called from the asyncio thread.
        """
        self.all_cancel_requests.append(request)
        await asyncio.sleep(0)

    async def wait_for_requests(self, num_requests=1) -> Sequence[quantum.QuantumRunStreamRequest]:
        """Wait til `num_requests` number of requests are received via `quantum_run_stream()`.

        This must be called from the duet thread.

        Returns:
            The received requests.
        """
        requests = []
        for _ in range(num_requests):
            requests.append(await self._request_buffer.__anext__())
        return requests

    async def reply(self, response_or_exception: Union[quantum.QuantumRunStreamResponse, BaseException]):
        """Sends a response or raises an exception to the `quantum_run_stream()` caller.

        If input response is missing `message_id`, it is defaulted to the `message_id` of the most
        recent request. This is to support the most common use case of responding immediately after
        a request.

        Assumes that at least one request must have been submitted to the StreamManager.

        This must be called from the duet thread.
        """
        responses_and_exceptions = await self._responses_and_exceptions_future
        if isinstance(response_or_exception, quantum.QuantumRunStreamResponse) and (not response_or_exception.message_id):
            response_or_exception.message_id = self.all_stream_requests[-1].message_id

        async def send():
            await responses_and_exceptions.put(response_or_exception)
        await self._executor.submit(send)

    async def wait_for_request_iterator_stop(self):
        """Wait for the request iterator to stop.

        This must be called from a duet thread.
        """
        await self._request_iterator_stopped
        self._request_iterator_stopped = duet.AwaitableFuture()