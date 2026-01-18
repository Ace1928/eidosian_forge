from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_params_extract(self):
    p = {'Parameters.member.1.ParameterKey': 'foo', 'Parameters.member.1.ParameterValue': 'bar', 'Parameters.member.2.ParameterKey': 'blarg', 'Parameters.member.2.ParameterValue': 'wibble'}
    params = api_utils.extract_param_pairs(p, prefix='Parameters', keyname='ParameterKey', valuename='ParameterValue')
    self.assertEqual(2, len(params))
    self.assertIn('foo', params)
    self.assertEqual('bar', params['foo'])
    self.assertIn('blarg', params)
    self.assertEqual('wibble', params['blarg'])