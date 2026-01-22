from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _update_share_group(self, share_group, **kwargs):
    """Updates a share group.

        :param share_group: either ShareGroup object or text with its UUID
        :rtype: :class:`ShareGroup`
        """
    share_group_id = base.getid(share_group)
    url = RESOURCE_PATH % share_group_id
    if not kwargs:
        return self._get(url, RESOURCE_NAME)
    else:
        body = {RESOURCE_NAME: kwargs}
        return self._update(url, body, RESOURCE_NAME)