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
class StackResourceLimitTest(StackResourceBaseTest):
    scenarios = [('3_4_0', dict(root=3, templ=4, nested=0, max=10, error=False)), ('3_8_0', dict(root=3, templ=8, nested=0, max=10, error=True)), ('3_8_2', dict(root=3, templ=8, nested=2, max=10, error=True)), ('3_5_2', dict(root=3, templ=5, nested=2, max=10, error=False)), ('3_6_2', dict(root=3, templ=6, nested=2, max=10, error=True)), ('3_12_2', dict(root=3, templ=12, nested=2, max=10, error=True))]

    def setUp(self):
        super(StackResourceLimitTest, self).setUp()
        self.res = self.parent_resource

    def test_resource_limit(self):
        total_resources = self.root + self.nested
        parser.Stack.total_resources = mock.Mock(return_value=total_resources)
        cfg.CONF.set_default('max_resources_per_stack', self.max)
        templ = mock.MagicMock()
        templ.__getitem__.return_value = range(self.templ)
        templ.RESOURCES = 'Resources'
        if self.error:
            self.assertRaises(exception.RequestLimitExceeded, self.res._validate_nested_resources, templ)
        else:
            self.assertIsNone(self.res._validate_nested_resources(templ))