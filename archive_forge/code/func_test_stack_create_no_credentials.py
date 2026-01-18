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
def test_stack_create_no_credentials(self):
    cfg.CONF.set_default('deferred_auth_method', 'password')
    stack_name = 'test_stack_create_no_credentials'
    params = {'foo': 'bar'}
    template = '{ "Template": "data" }'
    stk = tools.get_stack(stack_name, self.ctx)
    stk['WebServer'].requires_deferred_auth = True
    mock_tmpl = self.patchobject(templatem, 'Template', return_value=stk.t)
    mock_env = self.patchobject(environment, 'Environment', return_value=stk.env)
    mock_stack = self.patchobject(stack, 'Stack', return_value=stk)
    ctx_no_pwd = utils.dummy_context(password=None)
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.create_stack, ctx_no_pwd, stack_name, template, params, None, {}, None)
    self.assertEqual(exception.MissingCredentialError, ex.exc_info[0])
    self.assertEqual('Missing required credential: X-Auth-Key', str(ex.exc_info[1]))
    mock_tmpl.assert_called_once_with(template, files=None)
    mock_env.assert_called_once_with(params)
    mock_stack.assert_called_once_with(ctx_no_pwd, stack_name, stk.t, owner_id=None, nested_depth=0, user_creds_id=None, stack_user_project_id=None, convergence=cfg.CONF.convergence_engine, parent_resource=None)
    mock_tmpl.reset_mock()
    mock_env.reset_mock()
    mock_stack.reset_mock()
    ctx_no_pwd = utils.dummy_context(password=None)
    ctx_no_user = utils.dummy_context(user=None)
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.create_stack, ctx_no_user, stack_name, template, params, None, {})
    self.assertEqual(exception.MissingCredentialError, ex.exc_info[0])
    self.assertEqual('Missing required credential: X-Auth-User', str(ex.exc_info[1]))
    mock_tmpl.assert_called_once_with(template, files=None)
    mock_env.assert_called_once_with(params)
    mock_stack.assert_called_once_with(ctx_no_user, stack_name, stk.t, owner_id=None, nested_depth=0, user_creds_id=None, stack_user_project_id=None, convergence=cfg.CONF.convergence_engine, parent_resource=None)