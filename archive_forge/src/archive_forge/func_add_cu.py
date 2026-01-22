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
def add_cu(self, cu, name):
    """Add CUDA source in a string to the link. The name of the source
        file should be specified in `name`."""
    with driver.get_active_context() as ac:
        dev = driver.get_device(ac.devnum)
        cc = dev.compute_capability
    ptx, log = nvrtc.compile(cu, name, cc)
    if config.DUMP_ASSEMBLY:
        print(('ASSEMBLY %s' % name).center(80, '-'))
        print(ptx)
        print('=' * 80)
    ptx_name = os.path.splitext(name)[0] + '.ptx'
    self.add_ptx(ptx.encode(), ptx_name)