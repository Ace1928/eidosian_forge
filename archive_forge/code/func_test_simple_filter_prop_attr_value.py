import copy
from osc_lib.api import api
from osc_lib.api import utils as api_utils
from osc_lib.tests.api import fakes as api_fakes
def test_simple_filter_prop_attr_value(self):
    output = api_utils.simple_filter(copy.deepcopy(self.input_list), attr='b', value=2, property_field='props')
    self.assertEqual([api_fakes.RESP_ITEM_1, api_fakes.RESP_ITEM_2], output)
    output = api_utils.simple_filter(copy.deepcopy(self.input_list), attr='b', value=9, property_field='props')
    self.assertEqual([], output)