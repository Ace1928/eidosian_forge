import builtins
import errno
import io
import locale
import os
import time
import signal
import sys
import threading
import warnings
import contextlib
from time import monotonic as _time
import types
def _stdin_write(self, input):
    if input:
        try:
            self.stdin.write(input)
        except BrokenPipeError:
            pass
        except OSError as exc:
            if exc.errno == errno.EINVAL:
                pass
            else:
                raise
    try:
        self.stdin.close()
    except BrokenPipeError:
        pass
    except OSError as exc:
        if exc.errno == errno.EINVAL:
            pass
        else:
            raise