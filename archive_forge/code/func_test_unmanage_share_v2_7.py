from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
def test_unmanage_share_v2_7(self):
    share = 'fake_share'
    version = api_versions.APIVersion('2.7')
    mock_microversion = mock.Mock(api_version=version)
    manager = shares.ShareManager(api=mock_microversion)
    with mock.patch.object(manager, '_action', mock.Mock(return_value='fake')):
        result = manager.unmanage(share)
        manager._action.assert_called_once_with('unmanage', share)
        self.assertEqual('fake', result)