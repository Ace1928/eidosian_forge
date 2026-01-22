import multiprocessing
import multiprocessing.connection
import signal
import sys
import warnings
from typing import Optional
from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]
class ProcessExitedException(ProcessException):
    """Exception raised when a process failed due to signal or exited with a specific code."""
    __slots__ = ['exit_code']

    def __init__(self, msg: str, error_index: int, error_pid: int, exit_code: int, signal_name: Optional[str]=None):
        super().__init__(msg, error_index, error_pid)
        self.exit_code = exit_code
        self.signal_name = signal_name

    def __reduce__(self):
        return (type(self), (self.msg, self.error_index, self.pid, self.exit_code, self.signal_name))