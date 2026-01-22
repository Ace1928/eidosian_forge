import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def assertCorrectFloat64Atomics(self, kernel, shared=True):
    if config.ENABLE_CUDASIM:
        return
    asm = next(iter(kernel.inspect_asm().values()))
    if cc_X_or_above(6, 0):
        if cuda.runtime.get_version() > (12, 1):
            inst = 'red'
        else:
            inst = 'atom'
        if shared:
            inst = f'{inst}.shared'
        self.assertIn(f'{inst}.add.f64', asm)
    elif shared:
        self.assertIn('atom.shared.cas.b64', asm)
    else:
        self.assertIn('atom.cas.b64', asm)