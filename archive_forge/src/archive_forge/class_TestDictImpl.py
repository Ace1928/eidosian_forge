import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
class TestDictImpl(TestCase):

    def setUp(self):
        """Bind to the c_helper library and provide the ctypes wrapper.
        """
        dict_t = ctypes.c_void_p
        iter_t = ctypes.c_void_p
        hash_t = ctypes.c_ssize_t

        def wrap(name, restype, argtypes=()):
            proto = ctypes.CFUNCTYPE(restype, *argtypes)
            return proto(_helperlib.c_helpers[name])
        self.numba_test_dict = wrap('test_dict', ctypes.c_int)
        self.numba_dict_new_sized = wrap('dict_new_sized', ctypes.c_int, [ctypes.POINTER(dict_t), ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t])
        self.numba_dict_free = wrap('dict_free', None, [dict_t])
        self.numba_dict_length = wrap('dict_length', ctypes.c_ssize_t, [dict_t])
        self.numba_dict_insert_ez = wrap('dict_insert_ez', ctypes.c_int, [dict_t, ctypes.c_char_p, hash_t, ctypes.c_char_p])
        self.numba_dict_lookup = wrap('dict_lookup', ctypes.c_ssize_t, [dict_t, ctypes.c_char_p, hash_t, ctypes.c_char_p])
        self.numba_dict_delitem = wrap('dict_delitem', ctypes.c_int, [dict_t, hash_t, ctypes.c_ssize_t])
        self.numba_dict_popitem = wrap('dict_popitem', ctypes.c_int, [dict_t, ctypes.c_char_p, ctypes.c_char_p])
        self.numba_dict_iter_sizeof = wrap('dict_iter_sizeof', ctypes.c_size_t)
        self.numba_dict_iter = wrap('dict_iter', None, [iter_t, dict_t])
        self.numba_dict_iter_next = wrap('dict_iter_next', ctypes.c_int, [iter_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)])

    def test_simple_c_test(self):
        ret = self.numba_test_dict()
        self.assertEqual(ret, 0)

    def test_insertion_small(self):
        d = Dict(self, 4, 8)
        self.assertEqual(len(d), 0)
        self.assertIsNone(d.get('abcd'))
        d['abcd'] = 'beefcafe'
        self.assertEqual(len(d), 1)
        self.assertIsNotNone(d.get('abcd'))
        self.assertEqual(d['abcd'], 'beefcafe')
        d['abcd'] = 'cafe0000'
        self.assertEqual(len(d), 1)
        self.assertEqual(d['abcd'], 'cafe0000')
        d['abce'] = 'cafe0001'
        self.assertEqual(len(d), 2)
        self.assertEqual(d['abcd'], 'cafe0000')
        self.assertEqual(d['abce'], 'cafe0001')
        d['abcf'] = 'cafe0002'
        self.assertEqual(len(d), 3)
        self.assertEqual(d['abcd'], 'cafe0000')
        self.assertEqual(d['abce'], 'cafe0001')
        self.assertEqual(d['abcf'], 'cafe0002')

    def check_insertion_many(self, nmax):
        d = Dict(self, 8, 8)

        def make_key(v):
            return 'key_{:04}'.format(v)

        def make_val(v):
            return 'val_{:04}'.format(v)
        for i in range(nmax):
            d[make_key(i)] = make_val(i)
            self.assertEqual(len(d), i + 1)
        for i in range(nmax):
            self.assertEqual(d[make_key(i)], make_val(i))

    def test_insertion_many(self):
        self.check_insertion_many(nmax=7)
        self.check_insertion_many(nmax=8)
        self.check_insertion_many(nmax=9)
        self.check_insertion_many(nmax=31)
        self.check_insertion_many(nmax=32)
        self.check_insertion_many(nmax=33)
        self.check_insertion_many(nmax=1023)
        self.check_insertion_many(nmax=1024)
        self.check_insertion_many(nmax=1025)
        self.check_insertion_many(nmax=4095)
        self.check_insertion_many(nmax=4096)
        self.check_insertion_many(nmax=4097)

    def test_deletion_small(self):
        d = Dict(self, 4, 8)
        self.assertEqual(len(d), 0)
        self.assertIsNone(d.get('abcd'))
        d['abcd'] = 'cafe0000'
        d['abce'] = 'cafe0001'
        d['abcf'] = 'cafe0002'
        self.assertEqual(len(d), 3)
        self.assertEqual(d['abcd'], 'cafe0000')
        self.assertEqual(d['abce'], 'cafe0001')
        self.assertEqual(d['abcf'], 'cafe0002')
        self.assertEqual(len(d), 3)
        del d['abcd']
        self.assertIsNone(d.get('abcd'))
        self.assertEqual(d['abce'], 'cafe0001')
        self.assertEqual(d['abcf'], 'cafe0002')
        self.assertEqual(len(d), 2)
        with self.assertRaises(KeyError):
            del d['abcd']
        del d['abcf']
        self.assertIsNone(d.get('abcd'))
        self.assertEqual(d['abce'], 'cafe0001')
        self.assertIsNone(d.get('abcf'))
        self.assertEqual(len(d), 1)
        del d['abce']
        self.assertIsNone(d.get('abcd'))
        self.assertIsNone(d.get('abce'))
        self.assertIsNone(d.get('abcf'))
        self.assertEqual(len(d), 0)

    def check_delete_randomly(self, nmax, ndrop, nrefill, seed=0):
        random.seed(seed)
        d = Dict(self, 8, 8)
        keys = {}

        def make_key(v):
            return 'k_{:06x}'.format(v)

        def make_val(v):
            return 'v_{:06x}'.format(v)
        for i in range(nmax):
            d[make_key(i)] = make_val(i)
        for i in range(nmax):
            k = make_key(i)
            v = make_val(i)
            keys[k] = v
            self.assertEqual(d[k], v)
        self.assertEqual(len(d), nmax)
        droplist = random.sample(list(keys), ndrop)
        remain = keys.copy()
        for i, k in enumerate(droplist, start=1):
            del d[k]
            del remain[k]
            self.assertEqual(len(d), nmax - i)
        self.assertEqual(len(d), nmax - ndrop)
        for k in droplist:
            self.assertIsNone(d.get(k))
        for k in remain:
            self.assertEqual(d[k], remain[k])
        for i in range(nrefill):
            k = make_key(nmax + i)
            v = make_val(nmax + i)
            remain[k] = v
            d[k] = v
        self.assertEqual(len(remain), len(d))
        for k in remain:
            self.assertEqual(d[k], remain[k])

    def test_delete_randomly(self):
        self.check_delete_randomly(nmax=8, ndrop=2, nrefill=2)
        self.check_delete_randomly(nmax=13, ndrop=10, nrefill=31)
        self.check_delete_randomly(nmax=100, ndrop=50, nrefill=200)
        self.check_delete_randomly(nmax=100, ndrop=99, nrefill=100)
        self.check_delete_randomly(nmax=100, ndrop=100, nrefill=100)
        self.check_delete_randomly(nmax=1024, ndrop=999, nrefill=1)
        self.check_delete_randomly(nmax=1024, ndrop=999, nrefill=2048)

    def test_delete_randomly_large(self):
        self.check_delete_randomly(nmax=2 ** 17, ndrop=2 ** 16, nrefill=2 ** 10)

    def test_popitem(self):
        nmax = 10
        d = Dict(self, 8, 8)

        def make_key(v):
            return 'k_{:06x}'.format(v)

        def make_val(v):
            return 'v_{:06x}'.format(v)
        for i in range(nmax):
            d[make_key(i)] = make_val(i)
        self.assertEqual(len(d), nmax)
        k, v = d.popitem()
        self.assertEqual(len(d), nmax - 1)
        self.assertEqual(k, make_key(len(d)))
        self.assertEqual(v, make_val(len(d)))
        while len(d):
            n = len(d)
            k, v = d.popitem()
            self.assertEqual(len(d), n - 1)
            self.assertEqual(k, make_key(len(d)))
            self.assertEqual(v, make_val(len(d)))
        self.assertEqual(len(d), 0)
        with self.assertRaises(KeyError) as raises:
            d.popitem()
        self.assertIn('popitem(): dictionary is empty', str(raises.exception))

    def test_iter_items(self):
        d = Dict(self, 4, 4)
        nmax = 1000

        def make_key(v):
            return '{:04}'.format(v)

        def make_val(v):
            return '{:04}'.format(v + nmax)
        for i in range(nmax):
            d[make_key(i)] = make_val(i)
        for i, (k, v) in enumerate(d.items()):
            self.assertEqual(make_key(i), k)
            self.assertEqual(make_val(i), v)

    def check_sizing(self, key_size, val_size, nmax):
        d = Dict(self, key_size, val_size)

        def make_key(v):
            return '{:0{}}'.format(v, key_size)[:key_size]

        def make_val(v):
            return '{:0{}}'.format(nmax - v - 1, val_size)[:val_size]
        for i in range(nmax):
            d[make_key(i)] = make_val(i)
        for i, (k, v) in enumerate(d.items()):
            self.assertEqual(make_key(i), k)
            self.assertEqual(make_val(i), v)

    def test_sizing(self):
        for i in range(1, 8):
            self.check_sizing(key_size=i, val_size=i, nmax=2 ** i)

    def test_parameterized_types(self):
        """https://github.com/numba/numba/issues/6401"""
        register_model(ParametrizedType)(UniTupleModel)

        @typeof_impl.register(Parametrized)
        def typeof_unit(val, c):
            return ParametrizedType(val)

        @unbox(ParametrizedType)
        def unbox_parametrized(typ, obj, context):
            return context.unbox(types.UniTuple(typ.dtype, len(typ)), obj)

        def dict_vs_cache_vs_parametrized(v):
            assert 0

        @overload(dict_vs_cache_vs_parametrized)
        def ol_dict_vs_cache_vs_parametrized(v):
            typ = v

            def objmode_vs_cache_vs_parametrized_impl(v):
                d = typed.Dict.empty(types.unicode_type, typ)
                d['data'] = v
            return objmode_vs_cache_vs_parametrized_impl

        @jit(nopython=True, cache=True)
        def set_parametrized_data(x, y):
            dict_vs_cache_vs_parametrized(x)
            dict_vs_cache_vs_parametrized(y)
        x, y = (Parametrized(('a', 'b')), Parametrized(('a',)))
        set_parametrized_data(x, y)
        set_parametrized_data._make_finalizer()()
        set_parametrized_data._reset_overloads()
        set_parametrized_data.targetctx.init()
        for ii in range(50):
            self.assertIsNone(set_parametrized_data(x, y))