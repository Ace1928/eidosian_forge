from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _delete_share_group_snapshot(self, share_group_snapshot, force=False):
    """Delete a share group snapshot.

        :param share_group_snapshot: either ShareGroupSnapshot object or text
            with its UUID
        :param force: True to force the deletion
        """
    share_group_snapshot_id = base.getid(share_group_snapshot)
    if force:
        url = RESOURCE_PATH_ACTION % share_group_snapshot_id
        body = {'force_delete': None}
        self.api.client.post(url, body=body)
    else:
        url = RESOURCE_PATH % share_group_snapshot_id
        self._delete(url)