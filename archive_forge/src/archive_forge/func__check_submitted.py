import uuid
from concurrent.futures import ThreadPoolExecutor
from qiskit.providers import JobError, JobStatus
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from .base.base_primitive_job import BasePrimitiveJob, ResultT
def _check_submitted(self):
    if self._future is None:
        raise JobError('Primitive Job has not been submitted yet.')