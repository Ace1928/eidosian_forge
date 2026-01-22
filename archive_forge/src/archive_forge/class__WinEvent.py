import ctypes
import os
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
import platform
import traceback
import os, time, sys
class _WinEvent(object):

    def wait_for_event_set(self, timeout=None):
        """
            :param timeout: in seconds
            """
        if timeout is None:
            timeout = 4294967295
        else:
            timeout = int(timeout * 1000)
        ret = WaitForSingleObject(event, timeout)
        if ret in (0, 128):
            return True
        elif ret == 258:
            return False
        else:
            raise ctypes.WinError()