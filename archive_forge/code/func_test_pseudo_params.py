from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_pseudo_params(self):
    stack_name = 'test_stack'
    params = self.new_parameters(stack_name, {'Parameters': {}})
    self.assertEqual('test_stack', params['AWS::StackName'])
    self.assertEqual('arn:openstack:heat:::stacks/{0}/{1}'.format(stack_name, 'None'), params['AWS::StackId'])
    self.assertIn('AWS::Region', params)