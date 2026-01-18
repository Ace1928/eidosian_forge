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
def test_stack_update_exceeds_resource_limit(self):
    self.patchobject(context, 'StoredContext')
    stack_name = 'test_stack_update_exceeds_resource_limit'
    params = {}
    tpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'A': {'Type': 'GenericResourceType'}, 'B': {'Type': 'GenericResourceType'}, 'C': {'Type': 'GenericResourceType'}}}
    template = templatem.Template(tpl)
    old_stack = stack.Stack(self.ctx, stack_name, template)
    sid = old_stack.store()
    self.assertIsNotNone(sid)
    cfg.CONF.set_override('max_resources_per_stack', 2)
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.update_stack, self.ctx, old_stack.identifier(), tpl, params, None, {rpc_api.PARAM_CONVERGE: False})
    self.assertEqual(exception.RequestLimitExceeded, ex.exc_info[0])
    self.assertIn(exception.StackResourceLimitExceeded.msg_fmt, str(ex.exc_info[1]))