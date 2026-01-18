from heat.common import identifier
from heat.tests import common
def test_not_equal_dict(self):
    hi1 = identifier.HeatIdentifier('t', 's', 'i', 'p')
    hi2 = identifier.HeatIdentifier('t', 's', 'i', 'q')
    self.assertFalse(hi1 == dict(hi2))
    self.assertFalse(dict(hi1) == hi2)
    self.assertFalse(hi1 == {'tenant': 't', 'stack_name': 's', 'stack_id': 'i'})
    self.assertFalse({'tenant': 't', 'stack_name': 's', 'stack_id': 'i'} == hi1)