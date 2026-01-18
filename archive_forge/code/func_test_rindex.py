from itertools import product
from itertools import permutations
from numba import njit, typeof
from numba.core import types
import unittest
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.core.errors import TypingError, UnsupportedError
from numba.cpython.unicode import _MAX_UNICODE
from numba.core.types.functions import _header_lead
from numba.extending import overload
def test_rindex(self):
    pyfunc = rindex_usecase
    cfunc = njit(pyfunc)
    default_subs = [(s, ['', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES]
    cpython_subs = [('abcdefghiabc', ['', 'def', 'abc']), ('a' + 'Ă' * 100, ['a']), ('a' + '\U00100304' * 100, ['a']), ('Ă' + '\U00100304' * 100, ['Ă']), ('_a' + 'Ă' * 100, ['_a']), ('_a' + '\U00100304' * 100, ['_a']), ('_Ă' + '\U00100304' * 100, ['_Ă'])]
    for s, subs in default_subs + cpython_subs:
        for sub_str in subs:
            msg = 'Results "{}".rindex("{}") must be equal'
            self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str), msg=msg.format(s, sub_str))