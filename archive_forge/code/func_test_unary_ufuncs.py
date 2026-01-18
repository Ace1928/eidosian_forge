import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_unary_ufuncs(self):
    v = BlockVector(2)
    a = np.ones(3) * 0.5
    b = np.ones(2) * 0.8
    v.set_block(0, a)
    v.set_block(1, b)
    v2 = BlockVector(2)
    unary_funcs = [np.log10, np.sin, np.cos, np.exp, np.ceil, np.floor, np.tan, np.arctan, np.arcsin, np.arccos, np.sinh, np.cosh, np.abs, np.tanh, np.arcsinh, np.arctanh, np.fabs, np.sqrt, np.log, np.log2, np.absolute, np.isfinite, np.isinf, np.isnan, np.log1p, np.logical_not, np.exp2, np.expm1, np.sign, np.rint, np.square, np.positive, np.negative, np.rad2deg, np.deg2rad, np.conjugate, np.reciprocal]
    for fun in unary_funcs:
        v2.set_block(0, fun(v.get_block(0)))
        v2.set_block(1, fun(v.get_block(1)))
        res = fun(v)
        self.assertIsInstance(res, BlockVector)
        self.assertEqual(res.nblocks, 2)
        for i in range(2):
            self.assertTrue(np.allclose(res.get_block(i), v2.get_block(i)))
    other_funcs = [np.cumsum, np.cumprod, np.cumproduct]
    for fun in other_funcs:
        res = fun(v)
        self.assertIsInstance(res, BlockVector)
        self.assertEqual(res.nblocks, 2)
        self.assertTrue(np.allclose(fun(v.flatten()), res.flatten()))
    with self.assertRaises(Exception) as context:
        np.cbrt(v)