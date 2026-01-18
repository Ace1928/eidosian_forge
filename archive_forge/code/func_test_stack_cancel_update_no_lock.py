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
def test_stack_cancel_update_no_lock(self):
    stack_name = 'service_update_stack_test_cancel_same_engine'
    stk = tools.get_stack(stack_name, self.ctx)
    stk.state_set(stk.UPDATE, stk.IN_PROGRESS, 'test_override')
    stk.disable_rollback = False
    stk.store()
    self.patchobject(stack.Stack, 'load', return_value=stk)
    self.patchobject(stack_lock.StackLock, 'get_engine_id', return_value=None)
    self.patchobject(self.man.thread_group_mgr, 'send')
    self.man.stack_cancel_update(self.ctx, stk.identifier(), cancel_with_rollback=False)
    self.assertFalse(self.man.thread_group_mgr.send.called)