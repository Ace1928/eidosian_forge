import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
class DeviceOnlyEMMPlugin(cuda.HostOnlyCUDAMemoryManager):
    """
        Dummy EMM Plugin implementation for testing. It memorises which plugin
        API methods have been called so that the tests can check that Numba
        called into the plugin as expected.
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allocations = {}
        self.count = 0
        self.initialized = False
        self.memalloc_called = False
        self.reset_called = False
        self.get_memory_info_called = False
        self.get_ipc_handle_called = False

    def memalloc(self, size):
        if not self.initialized:
            raise RuntimeError('memalloc called before initialize')
        self.memalloc_called = True
        self.count += 1
        alloc_count = self.count
        self.allocations[alloc_count] = size
        finalizer_allocs = self.allocations

        def finalizer():
            del finalizer_allocs[alloc_count]
        ctx = weakref.proxy(self.context)
        ptr = ctypes.c_void_p(alloc_count)
        return cuda.cudadrv.driver.AutoFreePointer(ctx, ptr, size, finalizer=finalizer)

    def initialize(self):
        self.initialized = True

    def reset(self):
        self.reset_called = True

    def get_memory_info(self):
        self.get_memory_info_called = True
        return cuda.MemoryInfo(free=32, total=64)

    def get_ipc_handle(self, memory):
        self.get_ipc_handle_called = True
        return 'Dummy IPC handle for alloc %s' % memory.device_pointer.value

    @property
    def interface_version(self):
        return 1