from mmap import mmap
import errno
import os
import stat
import threading
import atexit
import tempfile
import time
import warnings
import weakref
from uuid import uuid4
from multiprocessing import util
from pickle import whichmodule, loads, dumps, HIGHEST_PROTOCOL, PicklingError
from .numpy_pickle import dump, load, load_temporary_memmap
from .backports import make_memmap
from .disk import delete_folder
from .externals.loky.backend import resource_tracker
def _get_backing_memmap(a):
    """Recursively look up the original np.memmap instance base if any."""
    b = getattr(a, 'base', None)
    if b is None:
        return None
    elif isinstance(b, mmap):
        return a
    else:
        return _get_backing_memmap(b)