from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _get_share_group_snapshot(self, share_group_snapshot):
    """Get a share group snapshot.

        :param share_group_snapshot: either share group snapshot object or text
            with its UUID
        :rtype: :class:`ShareGroupSnapshot`
        """
    share_group_snapshot_id = base.getid(share_group_snapshot)
    url = RESOURCE_PATH % share_group_snapshot_id
    return self._get(url, RESOURCE_NAME)