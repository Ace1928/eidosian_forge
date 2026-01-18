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
@mock.patch.object(stack.Stack, 'validate')
def test_stack_create_verify_err(self, mock_validate):
    mock_validate.side_effect = exception.StackValidationFailed(message='')
    stack_name = 'service_create_verify_err_test_stack'
    params = {'foo': 'bar'}
    template = '{ "Template": "data" }'
    stk = tools.get_stack(stack_name, self.ctx)
    mock_tmpl = self.patchobject(templatem, 'Template', return_value=stk.t)
    mock_env = self.patchobject(environment, 'Environment', return_value=stk.env)
    mock_stack = self.patchobject(stack, 'Stack', return_value=stk)
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.create_stack, self.ctx, stack_name, template, params, None, {})
    self.assertEqual(exception.StackValidationFailed, ex.exc_info[0])
    mock_tmpl.assert_called_once_with(template, files=None)
    mock_env.assert_called_once_with(params)
    mock_stack.assert_called_once_with(self.ctx, stack_name, stk.t, owner_id=None, nested_depth=0, user_creds_id=None, stack_user_project_id=None, convergence=cfg.CONF.convergence_engine, parent_resource=None)