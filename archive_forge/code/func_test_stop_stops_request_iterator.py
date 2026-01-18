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
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_stop_stops_request_iterator(self, client_constructor):
    expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
    fake_client, manager = setup(client_constructor)

    async def test():
        async with duet.timeout_scope(5):
            actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
            await fake_client.wait_for_requests()
            await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=fake_client.all_stream_requests[0].message_id, result=expected_result))
            await actual_result_future
            manager.stop()
            await fake_client.wait_for_request_iterator_stop()
    duet.run(test)