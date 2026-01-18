from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_range_max_schema(self):
    d = {'range': {'max': 10}, 'description': 'a range'}
    r = constraints.Range(max=10, description='a range')
    self.assertEqual(d, dict(r))