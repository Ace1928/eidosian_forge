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
def test_strip(self):
    STRIP_CASES = [('ass cii', 'ai'), ('ass cii', None), ('asscii', 'ai '), ('asscii ', 'ai '), (' asscii  ', 'ai '), (' asscii  ', 'asci '), (' asscii  ', 's'), ('      ', ' '), ('', ' '), ('', ''), ('', None), (' ', None), ('  asscii  ', 'ai '), ('  asscii  ', ''), ('  asscii  ', None), ('tú quién te crees?', 'étú? '), ('  tú quién te crees?   ', 'étú? '), ('  tú qrees?   ', ''), ('  tú quién te crees?   ', None), ('大处 着眼，小处着手。大大大处', '大处'), (' 大处大处  ', ''), ('\t\nabcd\t', '\ta'), (' 大处大处  ', None), ('\t abcd \t', None), ('\n abcd \n', None), ('\r abcd \r', None), ('\x0b abcd \x0b', None), ('\x0c abcd \x0c', None), ('\u2029abcd\u205f', None), ('\x85abcd\u2009', None)]
    for pyfunc, case_name in [(strip_usecase, 'strip'), (lstrip_usecase, 'lstrip'), (rstrip_usecase, 'rstrip')]:
        cfunc = njit(pyfunc)
        for string, chars in STRIP_CASES:
            self.assertEqual(pyfunc(string), cfunc(string), "'%s'.%s()?" % (string, case_name))
    for pyfunc, case_name in [(strip_usecase_chars, 'strip'), (lstrip_usecase_chars, 'lstrip'), (rstrip_usecase_chars, 'rstrip')]:
        cfunc = njit(pyfunc)
        sig1 = types.unicode_type(types.unicode_type, types.Optional(types.unicode_type))
        cfunc_optional = njit([sig1])(pyfunc)

        def try_compile_bad_optional(*args):
            bad = types.unicode_type(types.unicode_type, types.Optional(types.float64))
            njit([bad])(pyfunc)
        for fn in (cfunc, try_compile_bad_optional):
            with self.assertRaises(TypingError) as raises:
                fn('tú quis?', 1.1)
            self.assertIn('The arg must be a UnicodeType or None', str(raises.exception))
        for fn in (cfunc, cfunc_optional):
            for string, chars in STRIP_CASES:
                self.assertEqual(pyfunc(string, chars), fn(string, chars), "'%s'.%s('%s')?" % (string, case_name, chars))