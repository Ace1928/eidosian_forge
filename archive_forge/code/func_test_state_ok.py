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
def test_state_ok(self):
    """Test case when check_create_complete should return True.

        check_create_complete should return True create task is
        done and the nested stack is in (<action>,COMPLETE) state.
        """
    self.mock_lock = self.patchobject(stack_lock.StackLock, 'get_engine_id')
    self.mock_lock.return_value = None
    self.status[1] = 'COMPLETE'
    complete = getattr(self.parent_resource, 'check_%s_complete' % self.action)
    self.assertIs(True, complete(None))
    self.mock_status.assert_called_once_with(self.parent_resource.context, self.parent_resource.resource_id)
    self.mock_lock.assert_called_once_with(self.parent_resource.context, self.parent_resource.resource_id)