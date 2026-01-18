import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def test_update_data(self):
    m = self._make_model()
    data = ScalarData({m.var[:, 'A']: 0.5, m.var[:, 'B']: 2.0})
    new_data = ScalarData({m.var[:, 'A']: 0.1})
    data.update_data(new_data)
    self.assertEqual(data.get_data(), {pyo.ComponentUID(m.var[:, 'A']): 0.1, pyo.ComponentUID(m.var[:, 'B']): 2.0})
    new_data = {m.var[:, 'A']: 0.2}
    data.update_data(new_data)
    self.assertEqual(data.get_data(), {pyo.ComponentUID(m.var[:, 'A']): 0.2, pyo.ComponentUID(m.var[:, 'B']): 2.0})