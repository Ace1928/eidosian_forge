import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
class TestGenerators(MemoryLeakMixin, TestCase):

    def check_generator(self, pygen, cgen):
        self.assertEqual(next(cgen), next(pygen))
        expected = [x for x in pygen]
        got = [x for x in cgen]
        self.assertEqual(expected, got)
        with self.assertRaises(StopIteration):
            next(cgen)

    def check_gen1(self, **kwargs):
        pyfunc = gen1
        cr = jit((types.int32,), **kwargs)(pyfunc)
        pygen = pyfunc(8)
        cgen = cr(8)
        self.check_generator(pygen, cgen)

    def test_gen1(self):
        self.check_gen1(**nopython_flags)

    def test_gen1_objmode(self):
        self.check_gen1(**forceobj_flags)

    def check_gen2(self, **kwargs):
        pyfunc = gen2
        cr = jit((types.int32,), **kwargs)(pyfunc)
        pygen = pyfunc(8)
        cgen = cr(8)
        self.check_generator(pygen, cgen)

    def test_gen2(self):
        self.check_gen2(**nopython_flags)

    def test_gen2_objmode(self):
        self.check_gen2(**forceobj_flags)

    def check_gen3(self, **kwargs):
        pyfunc = gen3
        cr = jit((types.int32,), **kwargs)(pyfunc)
        pygen = pyfunc(8)
        cgen = cr(8)
        self.check_generator(pygen, cgen)

    def test_gen3(self):
        self.check_gen3(**nopython_flags)

    def test_gen3_objmode(self):
        self.check_gen3(**forceobj_flags)

    def check_gen4(self, **kwargs):
        pyfunc = gen4
        cr = jit((types.int32,) * 3, **kwargs)(pyfunc)
        pygen = pyfunc(5, 6, 7)
        cgen = cr(5, 6, 7)
        self.check_generator(pygen, cgen)

    def test_gen4(self):
        self.check_gen4(**nopython_flags)

    def test_gen4_objmode(self):
        self.check_gen4(**forceobj_flags)

    def test_gen5(self):
        with self.assertTypingError() as raises:
            jit((), **nopython_flags)(gen5)
        self.assertIn('Cannot type generator: it does not yield any value', str(raises.exception))

    def test_gen5_objmode(self):
        cgen = jit((), **forceobj_flags)(gen5)()
        self.assertEqual(list(cgen), [])
        with self.assertRaises(StopIteration):
            next(cgen)

    def check_gen6(self, **kwargs):
        cr = jit((types.int32,) * 2, **kwargs)(gen6)
        cgen = cr(5, 6)
        l = []
        for i in range(3):
            l.append(next(cgen))
        self.assertEqual(l, [14] * 3)

    def test_gen6(self):
        self.check_gen6(**nopython_flags)

    def test_gen6_objmode(self):
        self.check_gen6(**forceobj_flags)

    def check_gen7(self, **kwargs):
        pyfunc = gen7
        cr = jit((types.Array(types.float64, 1, 'C'),), **kwargs)(pyfunc)
        arr = np.linspace(1, 10, 7)
        pygen = pyfunc(arr.copy())
        cgen = cr(arr)
        self.check_generator(pygen, cgen)

    def test_gen7(self):
        self.check_gen7(**nopython_flags)

    def test_gen7_objmode(self):
        self.check_gen7(**forceobj_flags)

    def check_gen8(self, **jit_args):
        pyfunc = gen8
        cfunc = jit(**jit_args)(pyfunc)

        def check(*args, **kwargs):
            self.check_generator(pyfunc(*args, **kwargs), cfunc(*args, **kwargs))
        check(2, 3)
        check(4)
        check(y=5)
        check(x=6, b=True)

    def test_gen8(self):
        self.check_gen8(nopython=True)

    def test_gen8_objmode(self):
        self.check_gen8(forceobj=True)

    def check_gen9(self, **kwargs):
        pyfunc = gen_bool
        cr = jit((), **kwargs)(pyfunc)
        pygen = pyfunc()
        cgen = cr()
        self.check_generator(pygen, cgen)

    def test_gen9(self):
        self.check_gen9(**nopython_flags)

    def test_gen9_objmode(self):
        self.check_gen9(**forceobj_flags)

    def check_consume_generator(self, gen_func):
        cgen = jit(nopython=True)(gen_func)
        cfunc = jit(nopython=True)(make_consumer(cgen))
        pyfunc = make_consumer(gen_func)
        expected = pyfunc(5)
        got = cfunc(5)
        self.assertPreciseEqual(got, expected)

    def test_consume_gen1(self):
        self.check_consume_generator(gen1)

    def test_consume_gen2(self):
        self.check_consume_generator(gen2)

    def test_consume_gen3(self):
        self.check_consume_generator(gen3)

    def check_ndindex(self, **kwargs):
        pyfunc = gen_ndindex
        cr = jit((types.UniTuple(types.intp, 2),), **kwargs)(pyfunc)
        shape = (2, 3)
        pygen = pyfunc(shape)
        cgen = cr(shape)
        self.check_generator(pygen, cgen)

    def test_ndindex(self):
        self.check_ndindex(**nopython_flags)

    def test_ndindex_objmode(self):
        self.check_ndindex(**forceobj_flags)

    def check_np_flat(self, pyfunc, **kwargs):
        cr = jit((types.Array(types.int32, 2, 'C'),), **kwargs)(pyfunc)
        arr = np.arange(6, dtype=np.int32).reshape((2, 3))
        self.check_generator(pyfunc(arr), cr(arr))
        crA = jit((types.Array(types.int32, 2, 'A'),), **kwargs)(pyfunc)
        arr = arr.T
        self.check_generator(pyfunc(arr), crA(arr))

    def test_np_flat(self):
        self.check_np_flat(gen_flat, **nopython_flags)

    def test_np_flat_objmode(self):
        self.check_np_flat(gen_flat, **forceobj_flags)

    def test_ndenumerate(self):
        self.check_np_flat(gen_ndenumerate, **nopython_flags)

    def test_ndenumerate_objmode(self):
        self.check_np_flat(gen_ndenumerate, **forceobj_flags)

    def test_type_unification_error(self):
        pyfunc = gen_unification_error
        with self.assertTypingError() as raises:
            jit((), **nopython_flags)(pyfunc)
        msg = "Can't unify yield type from the following types: complex128, none"
        self.assertIn(msg, str(raises.exception))

    def test_optional_expansion_type_unification_error(self):
        pyfunc = gen_optional_and_type_unification_error
        with self.assertTypingError() as raises:
            jit((), **nopython_flags)(pyfunc)
        msg = "Can't unify yield type from the following types: complex128, int%s, none"
        self.assertIn(msg % types.intp.bitwidth, str(raises.exception))

    def test_changing_tuple_type(self):
        pyfunc = gen_changing_tuple_type
        expected = list(pyfunc())
        got = list(njit(pyfunc)())
        self.assertEqual(expected, got)

    def test_changing_number_type(self):
        pyfunc = gen_changing_number_type
        expected = list(pyfunc())
        got = list(njit(pyfunc)())
        self.assertEqual(expected, got)