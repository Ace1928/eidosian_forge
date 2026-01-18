from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_reformat_dict_keys(self):
    keymap = {'foo': 'bar'}
    data = {'foo': 123}
    expected = {'bar': 123}
    result = api_utils.reformat_dict_keys(keymap, data)
    self.assertEqual(expected, result)