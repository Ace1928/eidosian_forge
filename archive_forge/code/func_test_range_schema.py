from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_range_schema(self):
    d = {'range': {'min': 5, 'max': 10}, 'description': 'a range'}
    r = constraints.Range(5, 10, description='a range')
    self.assertEqual(d, dict(r))