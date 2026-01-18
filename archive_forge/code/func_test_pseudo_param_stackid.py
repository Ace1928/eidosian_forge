from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_pseudo_param_stackid(self):
    stack_name = 'test_stack'
    params = self.new_parameters(stack_name, {'Parameters': {}}, stack_id='abc123')
    self.assertEqual('arn:openstack:heat:::stacks/{0}/{1}'.format(stack_name, 'abc123'), params['AWS::StackId'])
    stack_identifier = identifier.HeatIdentifier('', '', 'def456')
    params.set_stack_id(stack_identifier)
    self.assertEqual(stack_identifier.arn(), params['AWS::StackId'])