from contextlib import contextmanager
import numpy as np
from numba import cuda
from numba.cuda.testing import (unittest, skip_on_cudasim,
from numba.tests.support import captured_stderr
from numba.core import config
def test_pinned_memory(self):
    ctx = cuda.current_context()
    mem = ctx.memhostalloc(32)
    with self.check_ignored_exception(ctx):
        del mem