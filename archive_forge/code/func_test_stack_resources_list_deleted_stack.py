from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import identifier
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import dependencies
from heat.engine import resource as res
from heat.engine.resources.aws.ec2 import instance as ins
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@mock.patch.object(stack.Stack, 'load')
def test_stack_resources_list_deleted_stack(self, mock_load):
    stk = tools.setup_stack_with_mock(self, 'resource_list_deleted_stack', self.ctx)
    stack_id = stk.identifier()
    mock_load.return_value = stk
    tools.clean_up_stack(self, stk)
    resources = self.eng.list_stack_resources(self.ctx, stack_id)
    self.assertEqual(0, len(resources))