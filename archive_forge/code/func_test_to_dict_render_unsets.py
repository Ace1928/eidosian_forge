from copy import deepcopy
from oslo_utils import uuidutils
from octavia_lib.api.drivers import data_models
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
def test_to_dict_render_unsets(self):
    ref_lb_converted_to_dict = self.ref_lb.to_dict(render_unsets=True)
    new_ref_lib_dict = deepcopy(self.ref_lb_dict)
    new_ref_lib_dict['pools'] = None
    new_ref_lib_dict['listeners'] = None
    new_ref_lib_dict['additional_vips'] = None
    self.assertEqual(new_ref_lib_dict, ref_lb_converted_to_dict)