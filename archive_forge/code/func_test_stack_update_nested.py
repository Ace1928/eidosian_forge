from unittest import mock
import uuid
import eventlet.queue
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import messaging
from heat.common import service_utils
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine import environment
from heat.engine import resource
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_stack_update_nested(self):
    stack_name = 'service_update_nested_test_stack'
    parent_stack = tools.get_stack(stack_name + '_parent', self.ctx)
    owner_id = parent_stack.store()
    old_stack = tools.get_stack(stack_name, self.ctx, owner_id=owner_id, nested_depth=1, user_creds_id=parent_stack.user_creds_id)
    sid = old_stack.store()
    old_stack.set_stack_user_project_id('1234')
    s = stack_object.Stack.get_by_id(self.ctx, sid)
    stk = tools.get_stack(stack_name, self.ctx)
    tmpl_id = stk.t.store(self.ctx)
    mock_stack = self.patchobject(stack, 'Stack', return_value=stk)
    mock_load = self.patchobject(stack.Stack, 'load', return_value=old_stack)
    mock_tmpl = self.patchobject(templatem.Template, 'load', return_value=stk.t)
    mock_validate = self.patchobject(stk, 'validate', return_value=None)
    msgq_mock = mock.Mock()
    self.patchobject(eventlet.queue, 'LightQueue', side_effect=[msgq_mock, eventlet.queue.LightQueue()])
    api_args = {'timeout_mins': 60, rpc_api.PARAM_CONVERGE: False}
    result = self.man.update_stack(self.ctx, old_stack.identifier(), None, None, None, api_args, template_id=tmpl_id)
    self.assertEqual(old_stack.identifier(), result)
    self.assertIsInstance(result, dict)
    self.assertTrue(result['stack_id'])
    self.assertEqual([msgq_mock], self.man.thread_group_mgr.msg_queues)
    mock_tmpl.assert_called_once_with(self.ctx, tmpl_id)
    mock_stack.assert_called_once_with(self.ctx, stk.name, stk.t, convergence=False, current_traversal=old_stack.current_traversal, prev_raw_template_id=None, current_deps=None, disable_rollback=True, nested_depth=1, owner_id=owner_id, parent_resource=None, stack_user_project_id='1234', strict_validate=True, tenant_id='test_tenant_id', timeout_mins=60, user_creds_id=u'1', username='test_username', converge=False)
    mock_load.assert_called_once_with(self.ctx, stack=s, check_refresh_cred=True)
    mock_validate.assert_called_once_with()