from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_replicas
@ddt.data('reset_status', 'reset_replica_state')
def test_reset_state_actions(self, action):
    attr = 'status' if action == 'reset_status' else 'replica_state'
    method = getattr(self.manager, action.replace('status', 'state'))
    with mock.patch.object(self.manager, '_action', mock.Mock()):
        method(FAKE_REPLICA, 'some_status')
        self.manager._action.assert_called_once_with(action, FAKE_REPLICA, {attr: 'some_status'})