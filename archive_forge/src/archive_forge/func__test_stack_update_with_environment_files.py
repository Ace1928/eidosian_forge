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
def _test_stack_update_with_environment_files(self, stack_name, files_container=None):
    params = {}
    template = '{ "Template": "data" }'
    old_stack = tools.get_stack(stack_name, self.ctx)
    sid = old_stack.store()
    old_stack.set_stack_user_project_id('1234')
    stack_object.Stack.get_by_id(self.ctx, sid)
    stk = tools.get_stack(stack_name, self.ctx)
    self.patchobject(stack, 'Stack', return_value=stk)
    self.patchobject(stack.Stack, 'load', return_value=old_stack)
    self.patchobject(templatem, 'Template', return_value=stk.t)
    self.patchobject(environment, 'Environment', return_value=stk.env)
    self.patchobject(stk, 'validate', return_value=None)
    self.patchobject(eventlet.queue, 'LightQueue', side_effect=[mock.Mock(), eventlet.queue.LightQueue()])
    mock_merge = self.patchobject(env_util, 'merge_environments')
    files = None
    if files_container:
        files = {'/env/test.yaml': "{'resource_registry': {}}"}
    environment_files = ['env_1']
    self.man.update_stack(self.ctx, old_stack.identifier(), template, params, None, {rpc_api.PARAM_CONVERGE: False}, environment_files=environment_files, files_container=files_container)
    mock_merge.assert_called_once_with(environment_files, files, params, mock.ANY)