import json
import os
import signal
import socket
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from torch.distributed.elastic.utils.logging import get_logger
from .error_handler import ErrorHandler  # noqa: F401
from .handlers import get_error_handler  # noqa: F401
def _format_failure(self, idx: int, rank: int, failure: ProcessFailure) -> Tuple[str, int]:
    msg = failure.message
    if isinstance(failure.message, dict):
        msg = failure.message.get('extraInfo', {}).get('py_callstack', failure.message.get('message', '<N/A>')).replace('\n', '\n  ')
    fmt = Template(_FAILURE_FORMAT_TEMPLATE).substitute(idx=idx, time=failure.timestamp_isoformat(), hostname=socket.getfqdn(), rank=rank, local_rank=failure.local_rank, exitcode=failure.exitcode, pid=failure.pid, error_file=failure.error_file, message=msg)
    width = 0
    for line in fmt.split('\n'):
        width = max(width, len(line))
    return (fmt, width)