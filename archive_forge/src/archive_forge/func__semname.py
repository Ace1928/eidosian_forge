import errno
import sys
import tempfile
import threading
from . import context
from . import process
from . import util
from ._ext import _billiard, ensure_SemLock
from time import monotonic
def _semname(sl):
    try:
        return sl.name
    except AttributeError:
        pass