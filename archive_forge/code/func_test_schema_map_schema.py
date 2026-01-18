from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_schema_map_schema(self):
    d = {'type': 'map', 'description': 'A map', 'schema': {'Foo': {'type': 'string', 'description': 'A string', 'default': 'wibble', 'required': False, 'constraints': [{'length': {'min': 4, 'max': 8}}]}}, 'required': False}
    s = constraints.Schema(constraints.Schema.STRING, 'A string', default='wibble', constraints=[constraints.Length(4, 8)])
    m = constraints.Schema(constraints.Schema.MAP, 'A map', schema={'Foo': s})
    self.assertEqual(d, dict(m))