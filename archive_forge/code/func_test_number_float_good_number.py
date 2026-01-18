from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_number_float_good_number(self):
    schema = {'Type': 'Number', 'MinValue': '3.0', 'MaxValue': '4.0'}
    p = new_parameter('p', schema, 3.5)
    self.assertEqual(3.5, p.value())