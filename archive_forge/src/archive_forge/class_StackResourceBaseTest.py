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
class StackResourceBaseTest(common.HeatTestCase):

    def setUp(self):
        super(StackResourceBaseTest, self).setUp()
        self.ws_resname = 'provider_resource'
        self.empty_temp = templatem.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {self.ws_resname: ws_res_snippet}})
        self.ctx = utils.dummy_context()
        self.parent_stack = parser.Stack(self.ctx, 'test_stack', self.empty_temp, stack_id=str(uuid.uuid4()), user_creds_id='uc123', stack_user_project_id='aprojectid')
        resource_defns = self.empty_temp.resource_definitions(self.parent_stack)
        self.parent_resource = generic_rsrc.StackResourceType('test', resource_defns[self.ws_resname], self.parent_stack)