import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
class TestLargeConstDict(TestCase, MemoryLeakMixin):
    """
    gh #7894

    Tests that check a peephole optimization for constant
    dictionaries in Python 3.10. The bytecode changes when
    number of elements > 15, which splits the constant dictionary
    into multiple dictionaries that are joined by a DICT_UPDATE
    bytecode instruction.

    This optimization modifies the IR to rejoin dictionaries
    and remove the DICT_UPDATE generated code. This then allows
    code that depends on literal dictionaries or literal keys
    to succeed.
    """

    @skip_unless_py10_or_later
    def test_large_heterogeneous_const_dict(self):
        """
        Tests that a function with a large heterogeneous constant
        dictionary remains a constant.
        """

        def const_func():
            d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 'a'}
            return d['S']
        py_func = const_func
        cfunc = njit()(const_func)
        a = py_func()
        b = cfunc()
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_large_heterogeneous_LiteralStrKeyDict_literal_values(self):
        """Check the literal values for a LiteralStrKeyDict requiring
        optimizations because it is heterogeneous.
        """

        def bar(d):
            ...

        @overload(bar)
        def ol_bar(d):
            a = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 'a'}

            def specific_ty(z):
                return types.literal(z) if types.maybe_literal(z) else typeof(z)
            expected = {types.literal(x): specific_ty(y) for x, y in a.items()}
            self.assertTrue(isinstance(d, types.LiteralStrKeyDict))
            self.assertEqual(d.literal_value, expected)
            self.assertEqual(hasattr(d, 'initial_value'), False)
            return lambda d: d

        @njit
        def foo():
            d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 'a'}
            bar(d)
        foo()

    @skip_unless_py10_or_later
    def test_large_heterogeneous_const_keys_dict(self):
        """
        Tests that a function with a large heterogeneous constant
        dictionary remains a constant.
        """

        def const_keys_func(a):
            d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': a}
            return d['S']
        py_func = const_keys_func
        cfunc = njit()(const_keys_func)
        value = 'a_string'
        a = py_func(value)
        b = cfunc(value)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_large_dict_mutation_not_carried(self):
        """Checks that the optimization for large dictionaries
        do not incorrectly update initial values due to other
        mutations.
        """

        def bar(d):
            ...

        @overload(bar)
        def ol_bar(d):
            a = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 7}
            if d.initial_value is None:
                return lambda d: literally(d)
            self.assertTrue(isinstance(d, types.DictType))
            self.assertEqual(d.initial_value, a)
            return lambda d: d

        @njit
        def foo():
            d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 7}
            d['X'] = 4
            bar(d)
        foo()

    @skip_unless_py10_or_later
    def test_usercode_update_use_d2(self):
        """
        Tests an example using a regular update is
        not modified by the optimization.
        """

        def check_before(x):
            pass

        def check_after(x):
            pass
        checked_before = False
        checked_after = False

        @overload(check_before, prefer_literal=True)
        def ol_check_before(d):
            nonlocal checked_before
            if not checked_before:
                checked_before = True
                a = {'a': 1, 'b': 2, 'c': 3}
                self.assertTrue(isinstance(d, types.DictType))
                self.assertEqual(d.initial_value, a)
            return lambda d: None

        @overload(check_after, prefer_literal=True)
        def ol_check_after(d):
            nonlocal checked_after
            if not checked_after:
                checked_after = True
                self.assertTrue(isinstance(d, types.DictType))
                self.assertTrue(d.initial_value is None)
            return lambda d: None

        def const_dict_func():
            """
            Dictionary update between two constant
            dictionaries. This verifies d2 doesn't
            get incorrectly removed.
            """
            d1 = {'a': 1, 'b': 2, 'c': 3}
            d2 = {'d': 4, 'e': 4}
            check_before(d1)
            d1.update(d2)
            check_after(d1)
            if len(d1) > 4:
                return d2
            return d1
        py_func = const_dict_func
        cfunc = njit()(const_dict_func)
        a = py_func()
        b = cfunc()
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_large_const_dict_inline_controlflow(self):
        """
        Tests generating a large dictionary when one of
        the inputs requires inline control flow
        has the change suggested in the error message
        for inlined control flow.
        """

        def inline_func(a, flag):
            d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1 if flag else 2, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': a}
            return d['S']
        with self.assertRaises(UnsupportedError) as raises:
            njit()(inline_func)('a_string', False)
        self.assertIn('You can resolve this issue by moving the control flow out', str(raises.exception))

    @skip_unless_py10_or_later
    def test_large_const_dict_noninline_controlflow(self):
        """
        Tests generating large constant dict when one of the
        inputs has the change suggested in the error message
        for inlined control flow.
        """

        def non_inline_func(a, flag):
            val = 1 if flag else 2
            d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': val, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': a}
            return d['S']
        py_func = non_inline_func
        cfunc = njit()(non_inline_func)
        value = 'a_string'
        a = py_func(value, False)
        b = cfunc(value, False)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_fuse_twice_literal_values(self):
        """
        Tests that the correct literal values are generated
        for a dictionary that produces two DICT_UPDATE
        bytecode entries for the same dictionary.
        """

        def bar(d):
            ...

        @overload(bar)
        def ol_bar(d):
            a = {'a1': 1, 'a2': 2, 'a3': 3, 'a4': 4, 'a5': 5, 'a6': 6, 'a7': 7, 'a8': 8, 'a9': 9, 'a10': 10, 'a11': 11, 'a12': 12, 'a13': 13, 'a14': 14, 'a15': 15, 'a16': 16, 'a17': 17, 'a18': 18, 'a19': 19, 'a20': 20, 'a21': 21, 'a22': 22, 'a23': 23, 'a24': 24, 'a25': 25, 'a26': 26, 'a27': 27, 'a28': 28, 'a29': 29, 'a30': 30, 'a31': 31, 'a32': 32, 'a33': 33, 'a34': 34, 'a35': 35}
            if d.initial_value is None:
                return lambda d: literally(d)
            self.assertTrue(isinstance(d, types.DictType))
            self.assertEqual(d.initial_value, a)
            return lambda d: d

        @njit
        def foo():
            d = {'a1': 1, 'a2': 2, 'a3': 3, 'a4': 4, 'a5': 5, 'a6': 6, 'a7': 7, 'a8': 8, 'a9': 9, 'a10': 10, 'a11': 11, 'a12': 12, 'a13': 13, 'a14': 14, 'a15': 15, 'a16': 16, 'a17': 17, 'a18': 18, 'a19': 19, 'a20': 20, 'a21': 21, 'a22': 22, 'a23': 23, 'a24': 24, 'a25': 25, 'a26': 26, 'a27': 27, 'a28': 28, 'a29': 29, 'a30': 30, 'a31': 31, 'a32': 32, 'a33': 33, 'a34': 34, 'a35': 35}
            bar(d)
        foo()