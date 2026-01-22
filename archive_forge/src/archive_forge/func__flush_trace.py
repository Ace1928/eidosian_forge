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
def _flush_trace():
    global _trace_file
    if _trace_file:
        _trace_file.flush()