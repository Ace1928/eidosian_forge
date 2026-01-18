from copy import deepcopy
from oslo_utils import uuidutils
from octavia_lib.api.drivers import data_models
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
def test_to_dict_recursive(self):
    ref_lb_dict_with_listener = deepcopy(self.ref_lb_dict_with_listener)
    ref_lb_dict_with_listener['listeners'][0].pop('description', None)
    ref_lb_converted_to_dict = self.ref_lb.to_dict(recurse=True)
    self.assertEqual(ref_lb_dict_with_listener, ref_lb_converted_to_dict)