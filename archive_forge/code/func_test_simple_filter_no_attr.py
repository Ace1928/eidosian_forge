import copy
from osc_lib.api import api
from osc_lib.api import utils as api_utils
from osc_lib.tests.api import fakes as api_fakes
def test_simple_filter_no_attr(self):
    output = api_utils.simple_filter(copy.deepcopy(self.input_list))
    self.assertEqual(self.input_list, output)