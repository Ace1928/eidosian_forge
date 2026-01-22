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
def _cuda_python_active_blocks_per_multiprocessor(self, func, blocksize, memsize, flags):
    ps = [func.handle, blocksize, memsize]
    if not flags:
        return driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(*ps)
    ps.append(flags)
    return driver.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(*ps)