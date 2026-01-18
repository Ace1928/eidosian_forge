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
def test_resolve_attribute_dict(self):
    self.parent_resource.nested_identifier = mock.Mock()
    self.parent_resource.nested_identifier.return_value = {'foo': 'bar'}
    self.parent_resource._rpc_client = mock.MagicMock()
    output = {'outputs': [{'output_key': 'key', 'output_value': {'a': 1, 'b': 2}}]}
    self.parent_resource._rpc_client.show_stack.return_value = [output]
    self.assertEqual({'a': 1, 'b': 2}, self.parent_resource._resolve_attribute('key'))