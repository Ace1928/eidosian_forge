import asyncio
import functools
import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
class TestFailingStream(CUDATestCase):

    @unittest.skip
    @with_asyncio_loop
    async def test_failed_stream(self):
        ctx = cuda.current_context()
        module = ctx.create_module_ptx('\n            .version 6.5\n            .target sm_30\n            .address_size 64\n            .visible .entry failing_kernel() { trap; }\n        ')
        failing_kernel = module.get_function('failing_kernel')
        stream = cuda.stream()
        failing_kernel.configure((1,), (1,), stream=stream).__call__()
        done = stream.async_done()
        with self.assertRaises(Exception):
            await done
        self.assertIsNotNone(done.exception())