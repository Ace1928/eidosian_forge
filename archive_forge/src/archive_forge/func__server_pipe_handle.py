import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
def _server_pipe_handle(self, first):
    if self.closed():
        return None
    flags = _winapi.PIPE_ACCESS_DUPLEX | _winapi.FILE_FLAG_OVERLAPPED
    if first:
        flags |= _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
    h = _winapi.CreateNamedPipe(self._address, flags, _winapi.PIPE_TYPE_MESSAGE | _winapi.PIPE_READMODE_MESSAGE | _winapi.PIPE_WAIT, _winapi.PIPE_UNLIMITED_INSTANCES, windows_utils.BUFSIZE, windows_utils.BUFSIZE, _winapi.NMPWAIT_WAIT_FOREVER, _winapi.NULL)
    pipe = windows_utils.PipeHandle(h)
    self._free_instances.add(pipe)
    return pipe