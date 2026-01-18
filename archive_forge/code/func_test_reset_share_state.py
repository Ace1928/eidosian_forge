from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data(('2.6', 'os-reset_status'), ('2.7', 'reset_status'))
@ddt.unpack
def test_reset_share_state(self, microversion, action_name):
    state = 'available'
    share = 'fake_share'
    version = api_versions.APIVersion(microversion)
    mock_microversion = mock.Mock(api_version=version)
    manager = shares.ShareManager(api=mock_microversion)
    with mock.patch.object(manager, '_action', mock.Mock(return_value='fake')):
        result = manager.reset_state(share, state)
        manager._action.assert_called_once_with(action_name, share, {'status': state})
        self.assertEqual('fake', result)