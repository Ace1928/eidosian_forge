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
def test_stack_update_existing_parameters(self):
    stack_name = 'service_update_test_stack_existing_parameters'
    update_params = {'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'parameters': {'newparam': 123}, 'resource_registry': {'resources': {}}}
    api_args = {rpc_api.PARAM_TIMEOUT: 60, rpc_api.PARAM_EXISTING: True, rpc_api.PARAM_CONVERGE: False}
    t = template_format.parse(tools.wp_template)
    stk = tools.get_stack(stack_name, self.ctx, with_params=True)
    stk.store()
    stk.set_stack_user_project_id('1234')
    self.assertEqual({'KeyName': 'test'}, stk.t.env.params)
    t['parameters']['newparam'] = {'type': 'number'}
    with mock.patch('heat.engine.stack.Stack') as mock_stack:
        stk.update = mock.Mock()
        self.patchobject(service, 'NotifyEvent')
        mock_stack.load.return_value = stk
        mock_stack.validate.return_value = None
        result = self.man.update_stack(self.ctx, stk.identifier(), t, update_params, None, api_args)
        tmpl = mock_stack.call_args[0][2]
        self.assertEqual({'KeyName': 'test', 'newparam': 123}, tmpl.env.params)
        self.assertEqual(stk.identifier(), result)