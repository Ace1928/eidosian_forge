from heat.common import identifier
from heat.tests import common
def test_stack_path(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', 'p')
    self.assertEqual('s/i', hi.stack_path())