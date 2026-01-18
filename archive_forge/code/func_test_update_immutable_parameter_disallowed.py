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
def test_update_immutable_parameter_disallowed(self):
    template = '\nheat_template_version: 2014-10-16\nparameters:\n  param1:\n    type: string\n    immutable: true\n    default: foo\n'
    self.ctx = utils.dummy_context(password=None)
    stack_name = 'test_update_immutable_parameters'
    old_stack = tools.get_stack(stack_name, self.ctx, template=template)
    sid = old_stack.store()
    old_stack.set_stack_user_project_id('1234')
    s = stack_object.Stack.get_by_id(self.ctx, sid)
    self.patchobject(self.man, '_get_stack', return_value=s)
    self.patchobject(stack, 'Stack', return_value=old_stack)
    self.patchobject(stack.Stack, 'load', return_value=old_stack)
    params = {'param1': 'bar'}
    exc = self.assertRaises(dispatcher.ExpectedException, self.man.update_stack, self.ctx, old_stack.identifier(), old_stack.t.t, params, None, {rpc_api.PARAM_CONVERGE: False})
    self.assertEqual(exception.ImmutableParameterModified, exc.exc_info[0])
    self.assertEqual('The following parameters are immutable and may not be updated: param1', exc.exc_info[1].message)