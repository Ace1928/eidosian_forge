import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_setflags(self):
    v = self.ones
    with self.assertRaises(NotImplementedError) as ctx:
        vv = v.setflags()