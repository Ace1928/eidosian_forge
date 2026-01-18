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
def test_stack_update_existing_registry(self):
    stack_name = 'service_update_test_stack_existing_registry'
    intital_registry = {'OS::Foo': 'foo.yaml', 'OS::Foo2': 'foo2.yaml', 'resources': {'myserver': {'OS::Server': 'myserver.yaml'}}}
    intial_params = {'encrypted_param_names': [], 'parameter_defaults': {}, 'parameters': {}, 'event_sinks': [], 'resource_registry': intital_registry}
    initial_files = {'foo.yaml': 'foo', 'foo2.yaml': 'foo2', 'myserver.yaml': 'myserver'}
    update_registry = {'OS::Foo2': 'newfoo2.yaml', 'resources': {'myother': {'OS::Other': 'myother.yaml'}}}
    update_params = {'encrypted_param_names': [], 'parameter_defaults': {}, 'parameters': {}, 'resource_registry': update_registry}
    update_files = {'newfoo2.yaml': 'newfoo', 'myother.yaml': 'myother'}
    api_args = {rpc_api.PARAM_TIMEOUT: 60, rpc_api.PARAM_EXISTING: True, rpc_api.PARAM_CONVERGE: False}
    t = template_format.parse(tools.wp_template)
    stk = utils.parse_stack(t, stack_name=stack_name, params=intial_params, files=initial_files)
    stk.set_stack_user_project_id('1234')
    self.assertEqual(intial_params, stk.t.env.env_as_dict())
    expected_reg = {'OS::Foo': 'foo.yaml', 'OS::Foo2': 'newfoo2.yaml', 'resources': {'myother': {'OS::Other': 'myother.yaml'}, 'myserver': {'OS::Server': 'myserver.yaml'}}}
    expected_env = {'encrypted_param_names': [], 'parameter_defaults': {}, 'parameters': {}, 'event_sinks': [], 'resource_registry': expected_reg}
    expected_files = {'foo.yaml': 'foo', 'foo2.yaml': 'foo2', 'myserver.yaml': 'myserver', 'newfoo2.yaml': 'newfoo', 'myother.yaml': 'myother'}
    with mock.patch('heat.engine.stack.Stack') as mock_stack:
        stk.update = mock.Mock()
        self.patchobject(service, 'NotifyEvent')
        mock_stack.load.return_value = stk
        mock_stack.validate.return_value = None
        result = self.man.update_stack(self.ctx, stk.identifier(), t, update_params, update_files, api_args)
        tmpl = mock_stack.call_args[0][2]
        self.assertEqual(expected_env, tmpl.env.env_as_dict())
        self.assertEqual(expected_files, tmpl.files.files)
        self.assertEqual(stk.identifier(), result)