from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
def test_wait_for_events_failed(self):
    ws = FakeWebSocket([{'body': {'timestamp': '2014-01-06T16:14:23Z', 'payload': {'resource_action': 'CREATE', 'resource_status': 'IN_PROGRESS', 'resource_name': 'mystack', 'physical_resource_id': 'stackid1', 'stack_id': 'stackid1'}}}, {'body': {'timestamp': '2014-01-06T16:14:26Z', 'payload': {'resource_action': 'CREATE', 'resource_status': 'FAILED', 'resource_name': 'mystack', 'physical_resource_id': 'stackid1', 'stack_id': 'stackid1'}}}])
    stack_status, msg = event_utils.wait_for_events(ws, 'mystack')
    self.assertEqual('CREATE_FAILED', stack_status)
    self.assertEqual('\n Stack mystack CREATE_FAILED \n', msg)