from heat.common import identifier
from heat.tests import common
def test_resource_init_path(self):
    si = identifier.HeatIdentifier('t', 's', 'i')
    pi = identifier.ResourceIdentifier(resource_name='p', **si)
    ri = identifier.ResourceIdentifier(resource_name='r', **pi)
    self.assertEqual('/resources/p/resources/r', ri.path)