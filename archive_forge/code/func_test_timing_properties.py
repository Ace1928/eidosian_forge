import unittest
import string
import numpy as np
from numba import njit, jit, literal_unroll
from numba.core import event as ev
from numba.tests.support import TestCase, override_config
def test_timing_properties(self):
    a = tuple(string.ascii_lowercase)

    @njit
    def bar(x):
        acc = 0
        for i in literal_unroll(a):
            if i in {'1': x}:
                acc += 1
            else:
                acc += np.sqrt(x[0, 0])
        return (np.sin(x), acc)

    @njit
    def foo(x):
        return bar(np.zeros((x, x)))
    with override_config('LLVM_PASS_TIMINGS', True):
        foo(1)

    def get_timers(fn, prop):
        md = fn.get_metadata(fn.signatures[0])
        return md[prop]
    foo_timers = get_timers(foo, 'timers')
    bar_timers = get_timers(bar, 'timers')
    foo_llvm_timer = get_timers(foo, 'llvm_pass_timings')
    bar_llvm_timer = get_timers(bar, 'llvm_pass_timings')
    self.assertLess(bar_timers['llvm_lock'], foo_timers['llvm_lock'])
    self.assertLess(bar_timers['compiler_lock'], foo_timers['compiler_lock'])
    self.assertLess(foo_llvm_timer.get_total_time(), foo_timers['llvm_lock'])
    self.assertLess(bar_llvm_timer.get_total_time(), bar_timers['llvm_lock'])
    self.assertLess(foo_timers['llvm_lock'], foo_timers['compiler_lock'])
    self.assertLess(bar_timers['llvm_lock'], bar_timers['compiler_lock'])