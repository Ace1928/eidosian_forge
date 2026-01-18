from openstack import exceptions
from openstack.shared_file_system.v2 import share as _share
from openstack.tests.functional.shared_file_system import base
def test_revert_share_to_snapshot(self):
    self.user_cloud.share.revert_share_to_snapshot(self.SHARE_ID, self.SHARE_SNAPSHOT_ID)
    get_reverted_share = self.user_cloud.share.get_share(self.SHARE_ID)
    self.user_cloud.share.wait_for_status(get_reverted_share, status='available', failures=['error'], interval=5, wait=self._wait_for_timeout)
    self.assertIsNotNone(get_reverted_share.id)