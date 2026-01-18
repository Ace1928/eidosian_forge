from heat.common import identifier
from heat.tests import common
def test_resource_init_no_path(self):
    si = identifier.HeatIdentifier('t', 's', 'i')
    ri = identifier.ResourceIdentifier(resource_name='r', **si)
    self.assertEqual('/resources/r', ri.path)