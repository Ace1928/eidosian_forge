from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, TYPE_CHECKING
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.providers.exceptions import JobTimeoutError
from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
def wait_for_final_state(self, timeout: Optional[float]=None, wait: float=5, callback: Optional[Callable]=None) -> None:
    """Poll the job status until it progresses to a final state such as ``DONE`` or ``ERROR``.

        Args:
            timeout: Seconds to wait for the job. If ``None``, wait indefinitely.
            wait: Seconds between queries.
            callback: Callback function invoked after each query.
                The following positional arguments are provided to the callback function:

                * job_id: Job ID
                * job_status: Status of the job from the last query
                * job: This BaseJob instance

                Note: different subclass might provide different arguments to
                the callback function.

        Raises:
            JobTimeoutError: If the job does not reach a final state before the
                specified timeout.
        """
    if not self._async:
        return
    start_time = time.time()
    status = self.status()
    while status not in JOB_FINAL_STATES:
        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time >= timeout:
            raise JobTimeoutError(f'Timeout while waiting for job {self.job_id()}.')
        if callback:
            callback(self.job_id(), status, self)
        time.sleep(wait)
        status = self.status()
    return