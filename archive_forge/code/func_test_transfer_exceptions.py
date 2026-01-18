import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.interfaces.var_linker import DynamicVarLinker
def test_transfer_exceptions(self):
    m1, m2 = self._make_models()
    vars1 = [pyo.Reference(m1.var[:, 'A']), pyo.Reference(m1.var[:, 'B'])]
    vars2 = [m2.x1, m2.x2, m2.x3]
    msg = 'must be provided two lists.*of equal length'
    with self.assertRaisesRegex(ValueError, msg):
        linker = DynamicVarLinker(vars1, vars2)
    vars1 = [pyo.Reference(m1.var[:, 'A']), pyo.Reference(m1.var[:, 'B']), m1.input]
    vars2 = [m2.x1, m2.x2, m2.x3]
    linker = DynamicVarLinker(vars1, vars2)
    msg = 'Source time points were not provided'
    with self.assertRaisesRegex(RuntimeError, msg):
        linker.transfer(t_target=m2.time)
    msg = 'Target time points were not provided'
    with self.assertRaisesRegex(RuntimeError, msg):
        linker.transfer(t_source=m1.time.first())