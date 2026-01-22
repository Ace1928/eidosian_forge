from typing import AsyncIterator, Dict, Optional, Union
import asyncio
import duet
import google.api_core.exceptions as google_exceptions
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
def _to_create_job_request(create_program_and_job_request: quantum.QuantumRunStreamRequest) -> quantum.QuantumRunStreamRequest:
    """Converted the QuantumRunStreamRequest from a CreateQuantumProgramAndJobRequest to a
    CreateQuantumJobRequest.
    """
    program = create_program_and_job_request.create_quantum_program_and_job.quantum_program
    job = create_program_and_job_request.create_quantum_program_and_job.quantum_job
    return quantum.QuantumRunStreamRequest(parent=create_program_and_job_request.parent, create_quantum_job=quantum.CreateQuantumJobRequest(parent=program.name, quantum_job=job))