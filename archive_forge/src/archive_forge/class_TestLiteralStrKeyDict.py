import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
class TestLiteralStrKeyDict(MemoryLeakMixin, TestCase):
    """ Tests for dictionaries with string keys that can map to anything!"""

    def test_basic_const_lowering_boxing(self):

        @njit
        def foo():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            return (ld['a'], ld['b'], ld['c'])
        self.assertEqual(foo(), (1, 2j, 'd'))

    def test_basic_nonconst_in_scope(self):

        @njit
        def foo(x):
            y = x + 5
            e = True if y > 2 else False
            ld = {'a': 1, 'b': 2j, 'c': 'd', 'non_const': e}
            return ld['non_const']
        self.assertTrue(foo(34))
        self.assertFalse(foo(-100))

    def test_basic_nonconst_freevar(self):
        e = 5

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            self.assertEqual(x.literal_value, {types.literal('a'): types.literal(1), types.literal('b'): typeof(2j), types.literal('c'): types.literal('d'), types.literal('d'): types.literal(5)})

            def impl(x):
                pass
            return impl

        @njit
        def foo():
            ld = {'a': 1, 'b': 2j, 'c': 'd', 'd': e}
            bar(ld)
        foo()

    def test_literal_value(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            self.assertEqual(x.literal_value, {types.literal('a'): types.literal(1), types.literal('b'): typeof(2j), types.literal('c'): types.literal('d')})

            def impl(x):
                pass
            return impl

        @njit
        def foo():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            bar(ld)
        foo()

    def test_list_and_array_as_value(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            self.assertEqual(x.literal_value, {types.literal('a'): types.literal(1), types.literal('b'): types.List(types.intp, initial_value=[1, 2, 3]), types.literal('c'): typeof(np.zeros(5))})

            def impl(x):
                pass
            return impl

        @njit
        def foo():
            b = [1, 2, 3]
            ld = {'a': 1, 'b': b, 'c': np.zeros(5)}
            bar(ld)
        foo()

    def test_repeated_key_literal_value(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            self.assertEqual(x.literal_value, {types.literal('a'): types.literal('aaaa'), types.literal('b'): typeof(2j), types.literal('c'): types.literal('d')})

            def impl(x):
                pass
            return impl

        @njit
        def foo():
            ld = {'a': 1, 'a': 10, 'b': 2j, 'c': 'd', 'a': 'aaaa'}
            bar(ld)
        foo()

    def test_read_only(self):

        def _len():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            return len(ld)

        def static_getitem():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            return ld['b']

        def contains():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            return ('b' in ld, 'f' in ld)

        def copy():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            new = ld.copy()
            return ld == new
        rdonlys = (_len, static_getitem, contains, copy)
        for test in rdonlys:
            with self.subTest(test.__name__):
                self.assertPreciseEqual(njit(test)(), test())

    def test_mutation_failure(self):

        def setitem():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld['a'] = 12

        def delitem():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            del ld['a']

        def popitem():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld.popitem()

        def pop():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld.pop()

        def clear():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld.clear()

        def setdefault():
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld.setdefault('f', 1)
        illegals = (setitem, delitem, popitem, pop, clear, setdefault)
        for test in illegals:
            with self.subTest(test.__name__):
                with self.assertRaises(TypingError) as raises:
                    njit(test)()
                expect = 'Cannot mutate a literal dictionary'
                self.assertIn(expect, str(raises.exception))

    def test_get(self):

        @njit
        def get(x):
            ld = {'a': 2j, 'c': 'd'}
            return ld.get(x)

        @njit
        def getitem(x):
            ld = {'a': 2j, 'c': 'd'}
            return ld[x]
        for test in (get, getitem):
            with self.subTest(test.__name__):
                with self.assertRaises(TypingError) as raises:
                    test('a')
                expect = 'Cannot get{item}() on a literal dictionary'
                self.assertIn(expect, str(raises.exception))

    def test_dict_keys(self):

        @njit
        def foo():
            ld = {'a': 2j, 'c': 'd'}
            return [x for x in ld.keys()]
        self.assertEqual(foo(), ['a', 'c'])

    def test_dict_values(self):

        @njit
        def foo():
            ld = {'a': 2j, 'c': 'd'}
            return ld.values()
        self.assertEqual(foo(), (2j, 'd'))

    def test_dict_items(self):

        @njit
        def foo():
            ld = {'a': 2j, 'c': 'd', 'f': np.zeros(5)}
            return ld.items()
        self.assertPreciseEqual(foo(), (('a', 2j), ('c', 'd'), ('f', np.zeros(5))))

    def test_dict_return(self):

        @njit
        def foo():
            ld = {'a': 2j, 'c': 'd'}
            return ld
        with self.assertRaises(TypeError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn('cannot convert native LiteralStrKey', excstr)

    def test_dict_unify(self):

        @njit
        def foo(x):
            if x + 7 > 4:
                a = {'a': 2j, 'c': 'd', 'e': np.zeros(4)}
            else:
                a = {'a': 5j, 'c': 'CAT', 'e': np.zeros((5,))}
            return a['c']
        self.assertEqual(foo(100), 'd')
        self.assertEqual(foo(-100), 'CAT')
        self.assertEqual(foo(100), foo.py_func(100))
        self.assertEqual(foo(-100), foo.py_func(-100))

    def test_dict_not_unify(self):

        @njit
        def key_mismatch(x):
            if x + 7 > 4:
                a = {'BAD_KEY': 2j, 'c': 'd', 'e': np.zeros(4)}
            else:
                a = {'a': 5j, 'c': 'CAT', 'e': np.zeros((5,))}
            py310_defeat1 = 1
            py310_defeat2 = 2
            py310_defeat3 = 3
            py310_defeat4 = 4
            return a['a']
        with self.assertRaises(TypingError) as raises:
            key_mismatch(100)
        self.assertIn('Cannot unify LiteralStrKey', str(raises.exception))

        @njit
        def value_type_mismatch(x):
            if x + 7 > 4:
                a = {'a': 2j, 'c': 'd', 'e': np.zeros((4, 3))}
            else:
                a = {'a': 5j, 'c': 'CAT', 'e': np.zeros((5,))}
            py310_defeat1 = 1
            py310_defeat2 = 2
            py310_defeat3 = 3
            py310_defeat4 = 4
            return a['a']
        with self.assertRaises(TypingError) as raises:
            value_type_mismatch(100)
        self.assertIn('Cannot unify LiteralStrKey', str(raises.exception))

    def test_dict_value_coercion(self):
        p = {(np.int32, np.int32): types.DictType, (np.int32, np.int8): types.DictType, (np.complex128, np.int32): types.DictType, (np.int32, np.complex128): types.LiteralStrKeyDict, (np.int32, np.array): types.LiteralStrKeyDict, (np.array, np.int32): types.LiteralStrKeyDict, (np.int8, np.int32): types.LiteralStrKeyDict, (np.int64, np.float64): types.LiteralStrKeyDict}

        def bar(x):
            pass
        for dts, container in p.items():

            @overload(bar)
            def ol_bar(x):
                self.assertTrue(isinstance(x, container))

                def impl(x):
                    pass
                return impl
            ty1, ty2 = dts

            @njit
            def foo():
                d = {'a': ty1(1), 'b': ty2(2)}
                bar(d)
            foo()

    def test_build_map_op_code(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):

            def impl(x):
                pass
            return impl

        @njit
        def foo():
            a = {'a': {'b1': 10, 'b2': 'string'}}
            bar(a)
        foo()

    def test_dict_as_arg(self):

        @njit
        def bar(fake_kwargs=None):
            if fake_kwargs is not None:
                fake_kwargs['d'][:] += 10

        @njit
        def foo():
            a = 1
            b = 2j
            c = 'string'
            d = np.zeros(3)
            e = {'a': a, 'b': b, 'c': c, 'd': d}
            bar(fake_kwargs=e)
            return e['d']
        np.testing.assert_allclose(foo(), np.ones(3) * 10)

    def test_dict_with_single_literallist_value(self):

        @njit
        def foo():
            z = {'A': [lambda a: 2 * a, 'B']}
            return z['A'][0](5)
        self.assertPreciseEqual(foo(), foo.py_func())

    def test_tuple_not_in_mro(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            self.assertFalse(isinstance(x, types.BaseTuple))
            self.assertTrue(isinstance(x, types.LiteralStrKeyDict))
            return lambda x: ...

        @njit
        def foo():
            d = {'a': 1, 'b': 'c'}
            bar(d)
        foo()

    def test_const_key_not_in_dict(self):

        @njit
        def foo():
            a = {'not_a': 2j, 'c': 'd', 'e': np.zeros(4)}
            return a['a']
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn("Key 'a' is not in dict.", str(raises.exception))

    def test_uncommon_identifiers(self):

        @njit
        def foo():
            d = {'0': np.ones(5), '1': 4}
            return len(d)
        self.assertPreciseEqual(foo(), foo.py_func())

        @njit
        def bar():
            d = {'+': np.ones(5), 'x--': 4}
            return len(d)
        self.assertPreciseEqual(bar(), bar.py_func())

    def test_update_error(self):

        @njit
        def foo():
            d1 = {'a': 2, 'b': 4, 'c': 'a'}
            d1.update({'x': 3})
            return d1
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('Cannot mutate a literal dictionary', str(raises.exception))