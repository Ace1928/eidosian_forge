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
@mock.patch.object(threadgroup, 'ThreadGroup')
@mock.patch.object(stack.Stack, 'validate')
def _test_stack_create(self, stack_name, mock_validate, mock_tg, environment_files=None, files_container=None, error=False):
    mock_tg.return_value = tools.DummyThreadGroup()
    params = {'foo': 'bar'}
    template = '{ "Template": "data" }'
    stk = tools.get_stack(stack_name, self.ctx, convergence=cfg.CONF.convergence_engine)
    files = None
    if files_container:
        files = {'/env/test.yaml': "{'resource_registry': {}}"}
    mock_tmpl = self.patchobject(templatem, 'Template', return_value=stk.t)
    mock_env = self.patchobject(environment, 'Environment', return_value=stk.env)
    mock_stack = self.patchobject(stack, 'Stack', return_value=stk)
    mock_merge = self.patchobject(env_util, 'merge_environments')
    if not error:
        result = self.man.create_stack(self.ctx, stack_name, template, params, None, {}, environment_files=environment_files, files_container=files_container)
        self.assertEqual(stk.identifier(), result)
        self.assertIsInstance(result, dict)
        self.assertTrue(result['stack_id'])
        mock_tmpl.assert_called_once_with(template, files=files)
        mock_env.assert_called_once_with(params)
        mock_stack.assert_called_once_with(self.ctx, stack_name, stk.t, owner_id=None, nested_depth=0, user_creds_id=None, stack_user_project_id=None, convergence=cfg.CONF.convergence_engine, parent_resource=None)
        if environment_files:
            mock_merge.assert_called_once_with(environment_files, files, params, mock.ANY)
        mock_validate.assert_called_once_with()
    else:
        ex = self.assertRaises(dispatcher.ExpectedException, self.man.create_stack, self.ctx, stack_name, template, params, None, {}, environment_files=environment_files, files_container=files_container)
        self.assertEqual(exception.NotFound, ex.exc_info[0])
        self.assertIn('Could not fetch files from container test_container, reason: error.', str(ex.exc_info[1]))