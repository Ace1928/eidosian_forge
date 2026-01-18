from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_snapshots as snapshots
def test_update_no_data(self):
    fake_share_group_snapshot = fake.ShareGroupSnapshot()
    mock_get = self.mock_object(self.manager, '_get', mock.Mock(return_value=fake_share_group_snapshot))
    mock_update = self.mock_object(self.manager, '_update', mock.Mock(return_value=fake_share_group_snapshot))
    update_args = {}
    result = self.manager.update(fake.ShareGroupSnapshot(), **update_args)
    self.assertIs(fake_share_group_snapshot, result)
    mock_get.assert_called_once_with(snapshots.RESOURCE_PATH % fake.ShareGroupSnapshot.id, snapshots.RESOURCE_NAME)
    self.assertFalse(mock_update.called)