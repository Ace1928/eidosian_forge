from unittest import mock
import ddt
from manilaclient import base
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_servers
def test_reset_task_state(self):
    share_server = 'fake_share_server'
    state = 'fake_state'
    returned = 'fake'
    with mock.patch.object(self.manager, '_action', mock.Mock(return_value=returned)):
        result = self.manager.reset_task_state(share_server, state)
        self.manager._action.assert_called_once_with('reset_task_state', share_server, {'task_state': state})
        self.assertEqual(returned, result)