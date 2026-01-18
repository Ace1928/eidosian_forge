import contextlib
import json
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import exceptions as msg_exceptions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources import stack_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@mock.patch('heat.engine.environment.get_child_environment')
@mock.patch.object(stack_resource.parser, 'Stack')
def test_preview_with_implemented_child_resource(self, mock_stack_class, mock_env_class):
    nested_stack = mock.Mock()
    mock_stack_class.return_value = nested_stack
    nested_stack.preview_resources.return_value = 'preview_nested_stack'
    mock_env_class.return_value = 'environment'
    template = templatem.Template(template_format.parse(param_template))
    parent_t = self.parent_stack.t
    resource_defns = parent_t.resource_definitions(self.parent_stack)
    parent_resource = MyImplementedStackResource('test', resource_defns[self.ws_resname], self.parent_stack)
    params = {'KeyName': 'test'}
    parent_resource.set_template(template, params)
    validation_mock = mock.Mock(return_value=None)
    parent_resource._validate_nested_resources = validation_mock
    result = parent_resource.preview()
    mock_env_class.assert_called_once_with(self.parent_stack.env, params, child_resource_name='test', item_to_remove=None)
    self.assertEqual('preview_nested_stack', result)
    mock_stack_class.assert_called_once_with(mock.ANY, 'test_stack-test', mock.ANY, timeout_mins=None, disable_rollback=True, parent_resource=parent_resource.name, owner_id=self.parent_stack.id, user_creds_id=self.parent_stack.user_creds_id, stack_user_project_id=self.parent_stack.stack_user_project_id, adopt_stack_data=None, nested_depth=1)