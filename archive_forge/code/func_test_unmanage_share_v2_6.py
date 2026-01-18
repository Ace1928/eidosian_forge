from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data(type('ShareUUID', (object,), {'uuid': '1234'}), type('ShareID', (object,), {'id': '1234'}), '1234')
def test_unmanage_share_v2_6(self, share):
    version = api_versions.APIVersion('2.6')
    mock_microversion = mock.Mock(api_version=version)
    manager = shares.ShareManager(api=mock_microversion)
    with mock.patch.object(manager, '_action', mock.Mock(return_value='fake')):
        result = manager.unmanage(share)
        self.assertFalse(manager._action.called)
        self.assertNotEqual('fake', result)
        self.assertEqual(manager.api.client.post.return_value, result)
        manager.api.client.post.assert_called_once_with('/os-share-unmanage/1234/unmanage')