import sys
import os
import ctypes
import weakref
import functools
import warnings
import logging
import threading
import asyncio
import pathlib
from itertools import product
from abc import ABCMeta, abstractmethod
from ctypes import (c_int, byref, c_size_t, c_char, c_char_p, addressof,
import contextlib
import importlib
import numpy as np
from collections import namedtuple, deque
from numba import mviewbuf
from numba.core import utils, serialize, config
from .error import CudaSupportError, CudaDriverError
from .drvapi import API_PROTOTYPES
from .drvapi import cu_occupancy_b2d_size, cu_stream_callback_pyobj, cu_uuid
from numba.cuda.cudadrv import enums, drvapi, nvrtc, _extras
class PinnedMemory(mviewbuf.MemAlloc):
    """A pointer to a pinned buffer on the host.

    :param context: The context in which the pointer was mapped.
    :type context: Context
    :param owner: The object owning the memory. For EMM plugin implementation,
                  this ca
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the buffer in bytes.
    :type size: int
    :param owner: An object owning the buffer that has been pinned. For EMM
                  plugin implementation, the default of ``None`` suffices for
                  memory allocated in ``memhostalloc`` - for ``mempin``, it
                  should be the owner passed in to the ``mempin`` method.
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """

    def __init__(self, context, pointer, size, owner=None, finalizer=None):
        self.context = context
        self.owned = owner
        self.size = size
        self.host_pointer = pointer
        self.is_managed = finalizer is not None
        self.handle = self.host_pointer
        self._buflen_ = self.size
        if USE_NV_BINDING:
            self._bufptr_ = self.host_pointer
        else:
            self._bufptr_ = self.host_pointer.value
        if finalizer is not None:
            weakref.finalize(self, finalizer)

    def own(self):
        return self