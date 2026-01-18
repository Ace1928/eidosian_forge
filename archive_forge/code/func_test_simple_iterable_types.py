import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_simple_iterable_types(self):
    """Test all simple iterables that a List can be constructed from."""

    def generate_function(line):
        context = {}
        code = dedent('\n                from numba.typed import List\n                def bar():\n                    {}\n                    return l\n                ').format(line)
        exec(code, context)
        return njit(context['bar'])
    for line in ('l = List([0, 1, 2])', 'l = List(range(3))', 'l = List(List([0, 1, 2]))', 'l = List((0, 1, 2))', 'l = List(set([0, 1, 2]))'):
        foo = generate_function(line)
        cf_received, py_received = (foo(), foo.py_func())
        for result in (cf_received, py_received):
            for i in range(3):
                self.assertEqual(i, result[i])