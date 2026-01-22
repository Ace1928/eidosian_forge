import errno
import functools
import fcntl
import os
import struct
import threading
from . import exceptions
from . import _error_translation as errors
from .bindings import libzfs_core
from ._constants import MAXNAMELEN
from .ctypes import int32_t
from ._nvlist import nvlist_in, nvlist_out
class LazyInit(object):

    def __init__(self, lib):
        self._lib = lib
        self._inited = False
        self._lock = threading.Lock()

    def __getattr__(self, name):
        if not self._inited:
            with self._lock:
                if not self._inited:
                    ret = self._lib.libzfs_core_init()
                    if ret != 0:
                        raise exceptions.ZFSInitializationFailed(ret)
                    self._inited = True
        return getattr(self._lib, name)