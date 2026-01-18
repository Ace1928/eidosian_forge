from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.engine.cfn import template as cfntemplate
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import template as hottemplate
from heat.engine import resource as res
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
@mock.patch.object(stack_object.Stack, 'get_all')
def test_stack_list_show_nested(self, mock_stack_get_all):
    self.eng.list_stacks(self.ctx, show_nested=True)
    mock_stack_get_all.assert_called_once_with(self.ctx, limit=mock.ANY, sort_keys=mock.ANY, marker=mock.ANY, sort_dir=mock.ANY, filters=mock.ANY, show_deleted=mock.ANY, show_nested=True, show_hidden=mock.ANY, tags=mock.ANY, tags_any=mock.ANY, not_tags=mock.ANY, not_tags_any=mock.ANY)