from copy import deepcopy
from oslo_utils import uuidutils
from octavia_lib.api.drivers import data_models
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
def test_to_dict_private_attrs(self):
    private_dict = {'_test': 'foo'}
    ref_lb_converted_to_dict = self.ref_lb.to_dict(**private_dict)
    self.assertEqual(self.ref_lb_dict, ref_lb_converted_to_dict)