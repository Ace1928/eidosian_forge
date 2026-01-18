from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_schema_invariance(self):
    params1 = self.new_parameters('test', params_schema, {'User': 'foo', 'Defaulted': 'wibble'})
    self.assertEqual('wibble', params1['Defaulted'])
    params2 = self.new_parameters('test', params_schema, {'User': 'foo'})
    self.assertEqual('foobar', params2['Defaulted'])