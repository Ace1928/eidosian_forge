import numpy as np
import os
import sys
import ctypes
import functools
from numba.core import config, serialize, sigutils, types, typing, utils
from numba.core.caching import Cache, CacheImpl
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaPerformanceWarning
from numba.core.typing.typeof import Purpose, typeof
from numba.cuda.api import get_current_device
from numba.cuda.args import wrap_arg
from numba.cuda.compiler import compile_cuda, CUDACompiler
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv.devices import get_context
from numba.cuda.descriptor import cuda_target
from numba.cuda.errors import (missing_launch_config_msg,
from numba.cuda import types as cuda_types
from numba import cuda
from numba import _dispatcher
from warnings import warn
class ForAll(object):

    def __init__(self, dispatcher, ntasks, tpb, stream, sharedmem):
        if ntasks < 0:
            raise ValueError("Can't create ForAll with negative task count: %s" % ntasks)
        self.dispatcher = dispatcher
        self.ntasks = ntasks
        self.thread_per_block = tpb
        self.stream = stream
        self.sharedmem = sharedmem

    def __call__(self, *args):
        if self.ntasks == 0:
            return
        if self.dispatcher.specialized:
            specialized = self.dispatcher
        else:
            specialized = self.dispatcher.specialize(*args)
        blockdim = self._compute_thread_per_block(specialized)
        griddim = (self.ntasks + blockdim - 1) // blockdim
        return specialized[griddim, blockdim, self.stream, self.sharedmem](*args)

    def _compute_thread_per_block(self, dispatcher):
        tpb = self.thread_per_block
        if tpb != 0:
            return tpb
        else:
            ctx = get_context()
            kernel = next(iter(dispatcher.overloads.values()))
            kwargs = dict(func=kernel._codelibrary.get_cufunc(), b2d_func=0, memsize=self.sharedmem, blocksizelimit=1024)
            _, tpb = ctx.get_max_potential_block_size(**kwargs)
            return tpb