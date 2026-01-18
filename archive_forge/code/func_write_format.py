import atexit
import contextlib
import functools
import inspect
import io
import os
import platform
import sys
import threading
import traceback
import debugpy
from debugpy.common import json, timestamp, util
def write_format(level, format_string, *args, **kwargs):
    if level != 'error' and level not in _levels:
        return
    try:
        text = format_string.format(*args, **kwargs)
    except Exception:
        reraise_exception()
    return write(level, text, kwargs.pop('_to_files', all))