from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_schema_all(self):
    d = {'type': 'string', 'description': 'A string', 'default': 'wibble', 'required': False, 'constraints': [{'length': {'min': 4, 'max': 8}}]}
    s = constraints.Schema(constraints.Schema.STRING, 'A string', default='wibble', constraints=[constraints.Length(4, 8)])
    self.assertEqual(d, dict(s))