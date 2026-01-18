from typing import AsyncIterator, Dict, Optional, Union
import asyncio
import duet
import google.api_core.exceptions as google_exceptions
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
The execution coroutine, an asyncio coroutine to manage the lifecycle of a job execution.

        This coroutine sends QuantumRunStream requests to the request iterator and receives
        responses from the ResponseDemux.

        It initially sends a CreateQuantumProgramAndJobRequest, and retries if there is a retryable
        error by sending another request. The exact request type depends on the error.

        There is one execution coroutine per running job submission.

        Args:
            request_queue: The queue used to send requests to the stream coroutine.
            project_name: The full project ID resource path associated with the job.
            program: The Quantum Engine program representing the circuit to be executed.
            job: The Quantum Engine job to be executed.

        Raises:
            concurrent.futures.CancelledError: if either the request is cancelled or the stream
                coroutine is cancelled.
            google.api_core.exceptions.GoogleAPICallError: if the stream breaks with a non-retryable
                error.
            ValueError: if the response is of a type which is not recognized by this client.
        