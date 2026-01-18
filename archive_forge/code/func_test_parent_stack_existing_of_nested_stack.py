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
def test_parent_stack_existing_of_nested_stack(self):
    parent_t = self.parent_stack.t
    resource_defns = parent_t.resource_definitions(self.parent_stack)
    stk_resource = MyImplementedStackResource('test', resource_defns[self.ws_resname], self.parent_stack)
    stk_resource.child_params = mock.Mock(return_value={})
    stk_resource.child_template = mock.Mock(return_value=templatem.Template(self.simple_template, stk_resource.child_params))
    stk_resource._validate_nested_resources = mock.Mock()
    nest_stack = stk_resource._parse_nested_stack('test_nest_stack', stk_resource.child_template(), stk_resource.child_params())
    self.assertEqual(self.parent_stack, nest_stack.parent_resource._stack())