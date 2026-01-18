from heat.common import identifier
from heat.tests import common
def test_stack_name_slash(self):
    self.assertRaises(ValueError, identifier.HeatIdentifier, 't', 's/s', 'i', 'p')