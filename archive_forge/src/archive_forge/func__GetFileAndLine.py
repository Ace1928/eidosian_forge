import logging as _logging
import os as _os
import sys as _sys
import _thread
import time as _time
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading
from tensorflow.python.util.tf_export import tf_export
def _GetFileAndLine():
    """Returns (filename, linenumber) for the stack frame."""
    code, f = _get_caller()
    if not code:
        return ('<unknown>', 0)
    return (code.co_filename, f.f_lineno)