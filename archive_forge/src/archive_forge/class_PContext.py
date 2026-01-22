import abc
import logging
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntFlag
from multiprocessing import synchronize
from types import FrameType
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record
from torch.distributed.elastic.multiprocessing.redirects import (
from torch.distributed.elastic.multiprocessing.tail_log import TailLog
class PContext(abc.ABC):
    """
    The base class that standardizes operations over a set of processes that are launched via different mechanisms.

    The name ``PContext`` is intentional to disambiguate with ``torch.multiprocessing.ProcessContext``.

    .. warning:: stdouts and stderrs should ALWAYS be a superset of
                 tee_stdouts and tee_stderrs (respectively) this is b/c
                 tee is implemented as a redirect + tail -f <stdout/stderr.log>
    """

    def __init__(self, name: str, entrypoint: Union[Callable, str], args: Dict[int, Tuple], envs: Dict[int, Dict[str, str]], stdouts: Dict[int, str], stderrs: Dict[int, str], tee_stdouts: Dict[int, str], tee_stderrs: Dict[int, str], error_files: Dict[int, str], log_line_prefixes: Optional[Dict[int, str]]=None):
        self.name = name
        nprocs = len(args)
        _validate_full_rank(stdouts, nprocs, 'stdouts')
        _validate_full_rank(stderrs, nprocs, 'stderrs')
        self.entrypoint = entrypoint
        self.args = args
        self.envs = envs
        self.stdouts = stdouts
        self.stderrs = stderrs
        self.error_files = error_files
        self.nprocs = nprocs
        self._stdout_tail = TailLog(name, tee_stdouts, sys.stdout, log_line_prefixes)
        self._stderr_tail = TailLog(name, tee_stderrs, sys.stderr, log_line_prefixes)

    def start(self) -> None:
        """Start processes using parameters defined in the constructor."""
        signal.signal(signal.SIGTERM, _terminate_process_handler)
        signal.signal(signal.SIGINT, _terminate_process_handler)
        if not IS_WINDOWS:
            signal.signal(signal.SIGHUP, _terminate_process_handler)
            signal.signal(signal.SIGQUIT, _terminate_process_handler)
        self._start()
        self._stdout_tail.start()
        self._stderr_tail.start()

    @abc.abstractmethod
    def _start(self) -> None:
        """Start processes using strategy defined in a particular context."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _poll(self) -> Optional[RunProcsResult]:
        """
        Poll the run status of the processes running under this context.
        This method follows an "all-or-nothing" policy and returns
        a ``RunProcessResults`` object if either all processes complete
        successfully or any process fails. Returns ``None`` if
        all processes are still running.
        """
        raise NotImplementedError()

    def wait(self, timeout: float=-1, period: float=1) -> Optional[RunProcsResult]:
        """
        Wait for the specified ``timeout`` seconds, polling every ``period`` seconds
        for the processes to be done. Returns ``None`` if the processes are still running
        on timeout expiry. Negative timeout values are interpreted as "wait-forever".
        A timeout value of zero simply queries the status of the processes (e.g. equivalent
        to a poll).

        ..note: Multiprocessing library registers SIGTERM and SIGINT signal handlers that raise
                ``SignalException`` when the signals received. It is up to the consumer of the code
                to properly handle the exception. It is important not to swallow the exception otherwise
                the process would not terminate. Example of the typical workflow can be:

        .. code-block:: python
            pc = start_processes(...)
            try:
                pc.wait(1)
                .. do some other work
            except SignalException as e:
                pc.shutdown(e.sigval, timeout=30)

        If SIGTERM or SIGINT occurs, the code above will try to shutdown child processes by propagating
        received signal. If child processes will not terminate in the timeout time, the process will send
        the SIGKILL.
        """
        if timeout == 0:
            return self._poll()
        if timeout < 0:
            timeout = sys.maxsize
        expiry = time.time() + timeout
        while time.time() < expiry:
            pr = self._poll()
            if pr:
                return pr
            time.sleep(period)
        return None

    @abc.abstractmethod
    def pids(self) -> Dict[int, int]:
        """Return pids of processes mapped by their respective local_ranks."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _close(self, death_sig: signal.Signals, timeout: int=30) -> None:
        """
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).
        """
        raise NotImplementedError()

    def close(self, death_sig: Optional[signal.Signals]=None, timeout: int=30) -> None:
        """
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).

        Args:
            death_sig: Death signal to terminate processes.
            timeout: Time to wait for processes to finish, if process is
                still alive after this time, it will be terminated via SIGKILL.
        """
        if not death_sig:
            death_sig = _get_default_signal()
        self._close(death_sig=death_sig, timeout=timeout)
        if self._stdout_tail:
            self._stdout_tail.stop()
        if self._stderr_tail:
            self._stderr_tail.stop()