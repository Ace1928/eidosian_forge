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
def test_stop_cancels_existing_sends(self, client_constructor):
    fake_client, manager = setup(client_constructor)

    async def test():
        async with duet.timeout_scope(5):
            actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
            await fake_client.wait_for_requests()
            manager.stop()
            with pytest.raises(concurrent.futures.CancelledError):
                await actual_result_future
    duet.run(test)