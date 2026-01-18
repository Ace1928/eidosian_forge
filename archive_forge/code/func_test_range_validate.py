from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_range_validate(self):
    r = constraints.Range(min=5, max=5, description='a range')
    r.validate(5)