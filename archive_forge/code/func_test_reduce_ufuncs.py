import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_reduce_ufuncs(self):
    v = BlockVector(2)
    a = np.ones(3) * 0.5
    b = np.ones(2) * 0.8
    v.set_block(0, a)
    v.set_block(1, b)
    reduce_funcs = [np.sum, np.max, np.min, np.prod, np.mean]
    for fun in reduce_funcs:
        self.assertAlmostEqual(fun(v), fun(v.flatten()))
    other_funcs = [np.all, np.any, np.std, np.ptp]
    for fun in other_funcs:
        self.assertAlmostEqual(fun(v), fun(v.flatten()))