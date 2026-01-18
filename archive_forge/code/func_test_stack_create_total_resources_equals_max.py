from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_service import threadgroup
from swiftclient import exceptions
from heat.common import environment_util as env_util
from heat.common import exception
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine import environment
from heat.engine import properties
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import service
from heat.engine import stack
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
@mock.patch.object(stack_object.Stack, 'count_total_resources')
def test_stack_create_total_resources_equals_max(self, ctr):
    stack_name = 'stack_create_total_resources_equals_max'
    params = {}
    tpl = {'heat_template_version': '2014-10-16', 'resources': {'A': {'type': 'GenericResourceType'}, 'B': {'type': 'GenericResourceType'}, 'C': {'type': 'GenericResourceType'}}}
    template = templatem.Template(tpl)
    stk = stack.Stack(self.ctx, stack_name, template)
    ctr.return_value = 3
    mock_tmpl = self.patchobject(templatem, 'Template', return_value=stk.t)
    mock_env = self.patchobject(environment, 'Environment', return_value=stk.env)
    mock_stack = self.patchobject(stack, 'Stack', return_value=stk)
    cfg.CONF.set_override('max_resources_per_stack', 3)
    result = self.man.create_stack(self.ctx, stack_name, template, params, None, {})
    mock_tmpl.assert_called_once_with(template, files=None)
    mock_env.assert_called_once_with(params)
    mock_stack.assert_called_once_with(self.ctx, stack_name, stk.t, owner_id=None, nested_depth=0, user_creds_id=None, stack_user_project_id=None, convergence=cfg.CONF.convergence_engine, parent_resource=None)
    self.assertEqual(stk.identifier(), result)
    root_stack_id = stk.root_stack_id()
    self.assertEqual(3, stk.total_resources(root_stack_id))
    stk.delete()