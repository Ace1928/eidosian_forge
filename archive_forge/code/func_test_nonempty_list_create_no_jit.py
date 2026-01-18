from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_nonempty_list_create_no_jit(self):
    with override_config('DISABLE_JIT', True):
        with forbid_codegen():
            l = List([1, 2, 3])
            self.assertEqual(type(l), list)
            self.assertEqual(l, [1, 2, 3])