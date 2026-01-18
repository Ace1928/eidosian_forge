from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.events as events
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_stack_index_event_id_uuid(self, mock_enforce):
    self._test_stack_index('a3455d8c-9f88-404d-a85b-5315293e67de', mock_enforce)