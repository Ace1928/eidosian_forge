import unittest
from itertools import product
from numba import types, njit, typed, errors
from numba.tests.support import TestCase
def test_static_getitem_on_type(self):

    def gen(numba_type, index):

        def foo():
            ty = numba_type[index]
            return typed.List.empty_list(ty)
        return foo
    tys = (types.bool_, types.float64, types.uint8, types.complex128)
    contig = slice(None, None, 1)
    noncontig = slice(None, None, None)
    indexes = (contig, noncontig, (noncontig, contig), (contig, noncontig), (noncontig, noncontig), (noncontig, noncontig, contig), (contig, noncontig, noncontig), (noncontig, noncontig, noncontig))
    for ty, idx in product(tys, indexes):
        compilable = njit(gen(ty, idx))
        expected = ty[idx]
        self.assertEqual(compilable()._dtype, expected)
        got = compilable.nopython_signatures[0].return_type.dtype
        self.assertEqual(got, expected)