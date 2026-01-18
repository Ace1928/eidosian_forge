from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_range_max_fail(self):
    r = constraints.Range(max=5, description='a range')
    self.assertRaises(ValueError, r.validate, 6)