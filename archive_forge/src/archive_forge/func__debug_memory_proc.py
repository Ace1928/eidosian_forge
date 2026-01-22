from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def _debug_memory_proc(message='', short=True):
    try:
        status_file = open('/proc/%s/status' % os.getpid(), 'rb')
    except OSError:
        return
    try:
        status = status_file.read()
    finally:
        status_file.close()
    if message:
        note(message)
    for line in status.splitlines():
        if not short:
            note(line)
        else:
            for field in _short_fields:
                if line.startswith(field):
                    note(line)
                    break