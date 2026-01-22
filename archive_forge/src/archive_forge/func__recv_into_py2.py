from __future__ import (absolute_import, division, print_function)
import functools
import io
import time
import traceback
from ansible.module_utils.six import PY2
def _recv_into_py2(self, buf, nbytes):
    err, data = win32file.ReadFile(self._handle, nbytes or len(buf))
    n = len(data)
    buf[:n] = data
    return n