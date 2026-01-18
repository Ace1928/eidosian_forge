from heat.common import identifier
from heat.tests import common
def test_path_components(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', 'p1/p2/p3')
    self.assertEqual(['p1', 'p2', 'p3'], hi._path_components())