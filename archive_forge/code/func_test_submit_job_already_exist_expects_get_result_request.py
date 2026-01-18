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
def test_submit_job_already_exist_expects_get_result_request(self, client_constructor):
    """Verifies the behavior when the client receives a JOB_ALREADY_EXISTS error.

        This error is only expected to be triggered in the following race condition:
        1. The client sends a CreateQuantumProgramAndJobRequest.
        2. The client's stream disconnects.
        3. The client retries with a new stream and a GetQuantumResultRequest.
        4. The job doesn't exist yet, and the client receives a "job not found" error.
        5. Scheduler creates the program and job.
        6. The client retries with a CreateJobRequest and fails with a "job already exists" error.

        The JOB_ALREADY_EXISTS error from `CreateQuantumJobRequest` is only possible if the job
        doesn't exist yet at the last `GetQuantumResultRequest`.
        """
    expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
    fake_client, manager = setup(client_constructor)

    async def test():
        async with duet.timeout_scope(5):
            actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
            await fake_client.wait_for_requests()
            await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
            await fake_client.wait_for_requests()
            await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)))
            await fake_client.wait_for_requests()
            await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.JOB_ALREADY_EXISTS)))
            await fake_client.wait_for_requests()
            await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
            actual_result = await actual_result_future
            manager.stop()
            assert actual_result == expected_result
            assert len(fake_client.all_stream_requests) == 4
            assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
            assert 'get_quantum_result' in fake_client.all_stream_requests[1]
            assert 'create_quantum_job' in fake_client.all_stream_requests[2]
            assert 'get_quantum_result' in fake_client.all_stream_requests[3]
    duet.run(test)