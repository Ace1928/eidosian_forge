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
def _log_and_unlink(filename):
    from .externals.loky.backend.resource_tracker import _resource_tracker
    util.debug('[FINALIZER CALL] object mapping to {} about to be deleted, decrementing the refcount of the file (pid: {})'.format(os.path.basename(filename), os.getpid()))
    _resource_tracker.maybe_unlink(filename, 'file')