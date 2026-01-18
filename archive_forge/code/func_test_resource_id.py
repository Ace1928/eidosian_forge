from heat.common import identifier
from heat.tests import common
def test_resource_id(self):
    ri = identifier.ResourceIdentifier('t', 's', 'i', '', 'r')
    self.assertEqual('r', ri.resource_name)