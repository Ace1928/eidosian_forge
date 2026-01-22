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
def _wait_cancel(self, event, done_callback):
    fut = self._wait_for_handle(event, None, True)
    fut._done_callback = done_callback
    return fut