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
class AutoFreePointer(MemoryPointer):
    """Modifies the ownership semantic of the MemoryPointer so that the
    instance lifetime is directly tied to the number of references.

    When the reference count reaches zero, the finalizer is invoked.

    Constructor arguments are the same as for :class:`MemoryPointer`.
    """

    def __init__(self, *args, **kwargs):
        super(AutoFreePointer, self).__init__(*args, **kwargs)
        self.refct -= 1