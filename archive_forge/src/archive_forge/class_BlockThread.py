from contextlib import contextmanager
import functools
import sys
import threading
import numpy as np
from .cudadrv.devicearray import FakeCUDAArray, FakeWithinKernelCUDAArray
from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
from ..errors import normalize_kernel_dimensions
from ..args import wrap_arg, ArgHint
class BlockThread(threading.Thread):
    """
    Manages the execution of a function for a single CUDA thread.
    """

    def __init__(self, f, manager, blockIdx, threadIdx, debug):
        if debug:

            def debug_wrapper(*args, **kwargs):
                np.seterr(divide='raise')
                f(*args, **kwargs)
            target = debug_wrapper
        else:
            target = f
        super(BlockThread, self).__init__(target=target)
        self.syncthreads_event = threading.Event()
        self.syncthreads_blocked = False
        self._manager = manager
        self.blockIdx = Dim3(*blockIdx)
        self.threadIdx = Dim3(*threadIdx)
        self.exception = None
        self.daemon = True
        self.abort = False
        self.debug = debug
        blockDim = Dim3(*self._manager._block_dim)
        self.thread_id = self.threadIdx.x + blockDim.x * (self.threadIdx.y + blockDim.y * self.threadIdx.z)

    def run(self):
        try:
            super(BlockThread, self).run()
        except Exception as e:
            tid = 'tid=%s' % list(self.threadIdx)
            ctaid = 'ctaid=%s' % list(self.blockIdx)
            if str(e) == '':
                msg = '%s %s' % (tid, ctaid)
            else:
                msg = '%s %s: %s' % (tid, ctaid, e)
            tb = sys.exc_info()[2]
            self.exception = (type(e)(msg), tb)

    def syncthreads(self):
        if self.abort:
            raise RuntimeError('abort flag set on syncthreads call')
        self.syncthreads_blocked = True
        self.syncthreads_event.wait()
        self.syncthreads_event.clear()
        if self.abort:
            raise RuntimeError('abort flag set on syncthreads clear')

    def syncthreads_count(self, value):
        idx = (self.threadIdx.x, self.threadIdx.y, self.threadIdx.z)
        self._manager.block_state[idx] = value
        self.syncthreads()
        count = np.count_nonzero(self._manager.block_state)
        self.syncthreads()
        return count

    def syncthreads_and(self, value):
        idx = (self.threadIdx.x, self.threadIdx.y, self.threadIdx.z)
        self._manager.block_state[idx] = value
        self.syncthreads()
        test = np.all(self._manager.block_state)
        self.syncthreads()
        return 1 if test else 0

    def syncthreads_or(self, value):
        idx = (self.threadIdx.x, self.threadIdx.y, self.threadIdx.z)
        self._manager.block_state[idx] = value
        self.syncthreads()
        test = np.any(self._manager.block_state)
        self.syncthreads()
        return 1 if test else 0

    def __str__(self):
        return 'Thread <<<%s, %s>>>' % (self.blockIdx, self.threadIdx)