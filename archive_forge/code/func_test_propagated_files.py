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
def test_propagated_files(self):
    """Test passing of the files map in the top level to the child.

        Makes sure that the files map in the top level stack are passed on to
        the child stack.
        """
    self.parent_stack.t.files['foo'] = 'bar'
    parsed_t = self.parent_resource._parse_child_template(self.templ, None)
    self.assertEqual({'foo': 'bar'}, parsed_t.files.files)