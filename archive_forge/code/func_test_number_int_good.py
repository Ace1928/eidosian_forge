from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_number_int_good(self):
    schema = {'Type': 'Number', 'MinValue': '3', 'MaxValue': '3'}
    p = new_parameter('p', schema, '3')
    self.assertEqual(3, p.value())