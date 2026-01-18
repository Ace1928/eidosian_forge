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
def test_submit_twice_in_parallel_expect_result_responses(self, client_constructor):
    expected_result0 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
    expected_result1 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job1')
    fake_client, manager = setup(client_constructor)

    async def test():
        async with duet.timeout_scope(5):
            actual_result0_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
            actual_result1_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB1)
            await fake_client.wait_for_requests(num_requests=2)
            await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=fake_client.all_stream_requests[0].message_id, result=expected_result0))
            await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=fake_client.all_stream_requests[1].message_id, result=expected_result1))
            actual_result1 = await actual_result1_future
            actual_result0 = await actual_result0_future
            manager.stop()
            assert actual_result0 == expected_result0
            assert actual_result1 == expected_result1
            assert len(fake_client.all_stream_requests) == 2
            assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
            assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[1]
    duet.run(test)