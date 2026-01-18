from heat.common import identifier
from heat.tests import common
def test_resource_name_slash(self):
    self.assertRaises(ValueError, identifier.ResourceIdentifier, 't', 's', 'i', 'p', 'r/r')