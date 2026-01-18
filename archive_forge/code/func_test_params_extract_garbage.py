from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_params_extract_garbage(self):
    p = {'Parameters.member.1.ParameterKey': 'foo', 'Parameters.member.1.ParameterValue': 'bar', 'Foo.1.ParameterKey': 'blarg', 'Foo.1.ParameterValue': 'wibble'}
    params = api_utils.extract_param_pairs(p, prefix='Parameters', keyname='ParameterKey', valuename='ParameterValue')
    self.assertEqual(1, len(params))
    self.assertIn('foo', params)
    self.assertEqual('bar', params['foo'])