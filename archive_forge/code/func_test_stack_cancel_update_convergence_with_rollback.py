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
@mock.patch('heat.engine.service.ThreadGroupManager', return_value=mock.Mock())
@mock.patch.object(stack_object.Stack, 'get_by_id')
@mock.patch.object(parser.Stack, 'load')
def test_stack_cancel_update_convergence_with_rollback(self, mock_load, mock_get_by_id, mock_tg):
    stk = mock.MagicMock()
    stk.id = 1
    stk.UPDATE = 'UPDATE'
    stk.IN_PROGRESS = 'IN_PROGRESS'
    stk.state = ('UPDATE', 'IN_PROGRESS')
    stk.status = stk.IN_PROGRESS
    stk.action = stk.UPDATE
    stk.convergence = True
    stk.rollback = mock.MagicMock(return_value=None)
    mock_load.return_value = stk
    self.patchobject(self.eng, '_get_stack')
    self.eng.thread_group_mgr.start = mock.MagicMock()
    self.eng.stack_cancel_update(self.ctx, 1, cancel_with_rollback=True)
    self.eng.thread_group_mgr.start.assert_called_once_with(1, stk.rollback)