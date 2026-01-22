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
def add_ptx(self, ptx, name='<cudapy-ptx>'):
    namebuf = name.encode('utf8')
    self._keep_alive += [ptx, namebuf]
    try:
        input_ptx = binding.CUjitInputType.CU_JIT_INPUT_PTX
        driver.cuLinkAddData(self.handle, input_ptx, ptx, len(ptx), namebuf, 0, [], [])
    except CudaAPIError as e:
        raise LinkerError('%s\n%s' % (e, self.error_log))