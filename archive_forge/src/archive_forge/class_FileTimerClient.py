import io
import json
import logging
import os
import select
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple
from torch.distributed.elastic.timer.api import TimerClient, TimerRequest
class FileTimerClient(TimerClient):
    """
    Client side of ``FileTimerServer``. This client is meant to be used
    on the same host that the ``FileTimerServer`` is running on and uses
    pid to uniquely identify a worker.
    This client uses a named_pipe to send timer requests to the
    ``FileTimerServer``. This client is a producer while the
    ``FileTimerServer`` is a consumer. Multiple clients can work with
    the same ``FileTimerServer``.

    Args:

        file_path: str, the path of a FIFO special file. ``FileTimerServer``
                        must have created it by calling os.mkfifo().

        signal: signal, the signal to use to kill the process. Using a
                        negative or zero signal will not kill the process.
    """

    def __init__(self, file_path: str, signal=signal.SIGKILL if sys.platform != 'win32' else signal.CTRL_C_EVENT) -> None:
        super().__init__()
        self._file_path = file_path
        self.signal = signal

    def _open_non_blocking(self) -> Optional[io.TextIOWrapper]:
        try:
            fd = os.open(self._file_path, os.O_WRONLY | os.O_NONBLOCK)
            return os.fdopen(fd, 'wt')
        except Exception:
            return None

    def _send_request(self, request: FileTimerRequest) -> None:
        file = self._open_non_blocking()
        if file is None:
            raise BrokenPipeError('Could not send the FileTimerRequest because FileTimerServer is not available.')
        with file:
            json_request = request.to_json()
            if len(json_request) > select.PIPE_BUF:
                raise RuntimeError(f'FileTimerRequest larger than {select.PIPE_BUF} bytes is not supported: {json_request}')
            file.write(json_request + '\n')

    def acquire(self, scope_id: str, expiration_time: float) -> None:
        self._send_request(request=FileTimerRequest(worker_pid=os.getpid(), scope_id=scope_id, expiration_time=expiration_time, signal=self.signal))

    def release(self, scope_id: str) -> None:
        self._send_request(request=FileTimerRequest(worker_pid=os.getpid(), scope_id=scope_id, expiration_time=-1, signal=0))