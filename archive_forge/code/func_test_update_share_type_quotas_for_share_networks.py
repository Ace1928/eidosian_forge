from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import quotas
def test_update_share_type_quotas_for_share_networks(self):
    manager = self._get_manager('2.39')
    with mock.patch.object(manager, '_update', mock.Mock(return_value='fake_delete')):
        self.assertRaises(ValueError, manager.update, 'fake_tenant_id', share_type='fake_share_type', share_networks=13)
        manager._update.assert_not_called()