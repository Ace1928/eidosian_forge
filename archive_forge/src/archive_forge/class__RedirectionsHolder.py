from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
class _RedirectionsHolder:
    _lock = ForkSafeLock(rlock=True)
    _stack_stdout = []
    _stack_stderr = []
    _pydevd_stdout_redirect_ = None
    _pydevd_stderr_redirect_ = None