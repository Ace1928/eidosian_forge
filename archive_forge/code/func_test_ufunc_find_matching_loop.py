import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def test_ufunc_find_matching_loop(self):
    f = numpy_support.ufunc_find_matching_loop
    np_add = FakeUFunc(_add_types)
    np_mul = FakeUFunc(_mul_types)
    np_isnan = FakeUFunc(_isnan_types)
    np_sqrt = FakeUFunc(_sqrt_types)

    def check(ufunc, input_types, sigs, output_types=()):
        """
            Check that ufunc_find_matching_loop() finds one of the given
            *sigs* for *ufunc*, *input_types* and optional *output_types*.
            """
        loop = f(ufunc, input_types + output_types)
        self.assertTrue(loop)
        if isinstance(sigs, str):
            sigs = (sigs,)
        self.assertIn(loop.ufunc_sig, sigs, 'inputs=%s and outputs=%s should have selected one of %s, got %s' % (input_types, output_types, sigs, loop.ufunc_sig))
        self.assertEqual(len(loop.numpy_inputs), len(loop.inputs))
        self.assertEqual(len(loop.numpy_outputs), len(loop.outputs))
        if not output_types:
            loop_explicit = f(ufunc, list(input_types) + loop.outputs)
            self.assertEqual(loop_explicit, loop)
        else:
            self.assertEqual(loop.outputs, list(output_types))
        loop_rt = f(ufunc, loop.inputs + loop.outputs)
        self.assertEqual(loop_rt, loop)
        return loop

    def check_exact(ufunc, input_types, sigs, output_types=()):
        """
            Like check(), but also ensure no casting of inputs occurred.
            """
        loop = check(ufunc, input_types, sigs, output_types)
        self.assertEqual(loop.inputs, list(input_types))

    def check_no_match(ufunc, input_types):
        loop = f(ufunc, input_types)
        self.assertIs(loop, None)
    check_exact(np_add, (types.bool_, types.bool_), '??->?')
    check_exact(np_add, (types.int8, types.int8), 'bb->b')
    check_exact(np_add, (types.uint8, types.uint8), 'BB->B')
    check_exact(np_add, (types.int64, types.int64), ('ll->l', 'qq->q'))
    check_exact(np_add, (types.uint64, types.uint64), ('LL->L', 'QQ->Q'))
    check_exact(np_add, (types.float32, types.float32), 'ff->f')
    check_exact(np_add, (types.float64, types.float64), 'dd->d')
    check_exact(np_add, (types.complex64, types.complex64), 'FF->F')
    check_exact(np_add, (types.complex128, types.complex128), 'DD->D')
    check_exact(np_add, (types.NPTimedelta('s'), types.NPTimedelta('s')), 'mm->m', output_types=(types.NPTimedelta('s'),))
    check_exact(np_add, (types.NPTimedelta('ms'), types.NPDatetime('s')), 'mM->M', output_types=(types.NPDatetime('ms'),))
    check_exact(np_add, (types.NPDatetime('s'), types.NPTimedelta('s')), 'Mm->M', output_types=(types.NPDatetime('s'),))
    check_exact(np_add, (types.NPDatetime('s'), types.NPTimedelta('')), 'Mm->M', output_types=(types.NPDatetime('s'),))
    check_exact(np_add, (types.NPDatetime('ns'), types.NPTimedelta('')), 'Mm->M', output_types=(types.NPDatetime('ns'),))
    check_exact(np_add, (types.NPTimedelta(''), types.NPDatetime('s')), 'mM->M', output_types=(types.NPDatetime('s'),))
    check_exact(np_add, (types.NPTimedelta(''), types.NPDatetime('ns')), 'mM->M', output_types=(types.NPDatetime('ns'),))
    check_exact(np_mul, (types.NPTimedelta('s'), types.int64), 'mq->m', output_types=(types.NPTimedelta('s'),))
    check_exact(np_mul, (types.float64, types.NPTimedelta('s')), 'dm->m', output_types=(types.NPTimedelta('s'),))
    check(np_add, (types.bool_, types.int8), 'bb->b')
    check(np_add, (types.uint8, types.bool_), 'BB->B')
    check(np_add, (types.int16, types.uint16), 'ii->i')
    check(np_add, (types.complex64, types.float64), 'DD->D')
    check(np_add, (types.float64, types.complex64), 'DD->D')
    int_types = [types.int32, types.uint32, types.int64, types.uint64]
    for intty in int_types:
        check(np_add, (types.float32, intty), 'ff->f')
        check(np_add, (types.float64, intty), 'dd->d')
        check(np_add, (types.complex64, intty), 'FF->F')
        check(np_add, (types.complex128, intty), 'DD->D')
    for intty in int_types:
        check(np_sqrt, (intty,), 'd->d')
        check(np_isnan, (intty,), 'd->?')
    check(np_mul, (types.NPTimedelta('s'), types.int32), 'mq->m', output_types=(types.NPTimedelta('s'),))
    check(np_mul, (types.NPTimedelta('s'), types.uint32), 'mq->m', output_types=(types.NPTimedelta('s'),))
    check(np_mul, (types.NPTimedelta('s'), types.float32), 'md->m', output_types=(types.NPTimedelta('s'),))
    check(np_mul, (types.float32, types.NPTimedelta('s')), 'dm->m', output_types=(types.NPTimedelta('s'),))
    check_no_match(np_add, (types.NPDatetime('s'), types.NPDatetime('s')))
    check_no_match(np_add, (types.NPTimedelta('s'), types.int64))