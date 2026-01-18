import asyncio
import functools
import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def with_asyncio_loop(f):

    @functools.wraps(f)
    def runner(*args, **kwds):
        loop = asyncio.new_event_loop()
        loop.set_debug(True)
        try:
            return loop.run_until_complete(f(*args, **kwds))
        finally:
            loop.close()
    return runner