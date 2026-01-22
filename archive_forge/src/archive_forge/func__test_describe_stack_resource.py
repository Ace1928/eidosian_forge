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
def _test_describe_stack_resource(self, mock_load):
    mock_load.return_value = self.stack
    self.patchobject(res.Resource, '_resolve_any_attribute', return_value=None)
    r = self.eng.describe_stack_resource(self.ctx, self.stack.identifier(), 'WebServer', with_attr=None)
    self.assertIn('resource_identity', r)
    self.assertIn('description', r)
    self.assertIn('updated_time', r)
    self.assertIn('stack_identity', r)
    self.assertIsNotNone(r['stack_identity'])
    self.assertIn('stack_name', r)
    self.assertEqual(self.stack.name, r['stack_name'])
    self.assertIn('metadata', r)
    self.assertIn('resource_status', r)
    self.assertIn('resource_status_reason', r)
    self.assertIn('resource_type', r)
    self.assertIn('physical_resource_id', r)
    self.assertIn('resource_name', r)
    self.assertIn('attributes', r)
    self.assertEqual('WebServer', r['resource_name'])
    mock_load.assert_called_once_with(self.ctx, stack=mock.ANY)