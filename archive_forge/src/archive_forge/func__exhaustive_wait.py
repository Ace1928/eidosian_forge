import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def _exhaustive_wait(handles, timeout):
    L = list(handles)
    ready = []
    while L:
        res = _winapi.WaitForMultipleObjects(L, False, timeout)
        if res == WAIT_TIMEOUT:
            break
        elif WAIT_OBJECT_0 <= res < WAIT_OBJECT_0 + len(L):
            res -= WAIT_OBJECT_0
        elif WAIT_ABANDONED_0 <= res < WAIT_ABANDONED_0 + len(L):
            res -= WAIT_ABANDONED_0
        else:
            raise RuntimeError('Should not get here')
        ready.append(L[res])
        L = L[res + 1:]
        timeout = 0
    return ready