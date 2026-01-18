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
def test_stack_update_existing_failed(self):
    """Update a stack using the same template doesn't work when FAILED."""
    stack_name = 'service_update_test_stack_existing_template'
    api_args = {rpc_api.PARAM_TIMEOUT: 60, rpc_api.PARAM_EXISTING: True, rpc_api.PARAM_CONVERGE: False}
    t = template_format.parse(tools.wp_template)
    self.man.thread_group_mgr = tools.DummyThreadGroupMgrLogStart()
    params = {}
    stack = utils.parse_stack(t, stack_name=stack_name, params=params)
    stack.set_stack_user_project_id('1234')
    self.assertEqual(t, stack.t.t)
    stack.action = stack.UPDATE
    stack.status = stack.FAILED
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.update_stack, self.ctx, stack.identifier(), None, params, None, api_args)
    self.assertEqual(exception.NotSupported, ex.exc_info[0])
    self.assertIn('PATCH update to non-COMPLETE stack', str(ex.exc_info[1]))