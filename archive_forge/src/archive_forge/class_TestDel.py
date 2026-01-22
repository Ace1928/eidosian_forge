from contextlib import contextmanager
import numpy as np
from numba import cuda
from numba.cuda.testing import (unittest, skip_on_cudasim,
from numba.tests.support import captured_stderr
from numba.core import config
@skip_on_cudasim('not supported on CUDASIM')
class TestDel(CUDATestCase):
    """
    Ensure resources are deleted properly without ignored exception.
    """

    @contextmanager
    def check_ignored_exception(self, ctx):
        with captured_stderr() as cap:
            yield
            ctx.deallocations.clear()
        self.assertFalse(cap.getvalue())

    def test_stream(self):
        ctx = cuda.current_context()
        stream = ctx.create_stream()
        with self.check_ignored_exception(ctx):
            del stream

    def test_event(self):
        ctx = cuda.current_context()
        event = ctx.create_event()
        with self.check_ignored_exception(ctx):
            del event

    def test_pinned_memory(self):
        ctx = cuda.current_context()
        mem = ctx.memhostalloc(32)
        with self.check_ignored_exception(ctx):
            del mem

    def test_mapped_memory(self):
        ctx = cuda.current_context()
        mem = ctx.memhostalloc(32, mapped=True)
        with self.check_ignored_exception(ctx):
            del mem

    def test_device_memory(self):
        ctx = cuda.current_context()
        mem = ctx.memalloc(32)
        with self.check_ignored_exception(ctx):
            del mem

    def test_managed_memory(self):
        ctx = cuda.current_context()
        mem = ctx.memallocmanaged(32)
        with self.check_ignored_exception(ctx):
            del mem

    def test_pinned_contextmanager(self):

        class PinnedException(Exception):
            pass
        arr = np.zeros(1)
        ctx = cuda.current_context()
        ctx.deallocations.clear()
        with self.check_ignored_exception(ctx):
            with cuda.pinned(arr):
                pass
            with cuda.pinned(arr):
                pass
            with cuda.defer_cleanup():
                with cuda.pinned(arr):
                    pass
                with cuda.pinned(arr):
                    pass
            try:
                with cuda.pinned(arr):
                    raise PinnedException
            except PinnedException:
                with cuda.pinned(arr):
                    pass

    def test_mapped_contextmanager(self):

        class MappedException(Exception):
            pass
        arr = np.zeros(1)
        ctx = cuda.current_context()
        ctx.deallocations.clear()
        with self.check_ignored_exception(ctx):
            with cuda.mapped(arr):
                pass
            with cuda.mapped(arr):
                pass
            with cuda.defer_cleanup():
                with cuda.mapped(arr):
                    pass
                with cuda.mapped(arr):
                    pass
            try:
                with cuda.mapped(arr):
                    raise MappedException
            except MappedException:
                with cuda.mapped(arr):
                    pass