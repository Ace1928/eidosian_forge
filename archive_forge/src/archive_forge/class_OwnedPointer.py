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
class OwnedPointer(object):

    def __init__(self, memptr, view=None):
        self._mem = memptr
        if view is None:
            self._view = self._mem
        else:
            assert not view.is_managed
            self._view = view
        mem = self._mem

        def deref():
            try:
                mem.refct -= 1
                assert mem.refct >= 0
                if mem.refct == 0:
                    mem.free()
            except ReferenceError:
                pass
        self._mem.refct += 1
        weakref.finalize(self, deref)

    def __getattr__(self, fname):
        """Proxy MemoryPointer methods
        """
        return getattr(self._view, fname)