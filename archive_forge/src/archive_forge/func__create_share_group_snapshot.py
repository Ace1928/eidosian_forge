from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _create_share_group_snapshot(self, share_group, name=None, description=None):
    """Create a share group snapshot.

        :param share_group: either ShareGroup object or text with its UUID
        :param name: text - name of the new group snapshot
        :param description: text - description of the group snapshot
        :rtype: :class:`ShareGroupSnapshot`
        """
    share_group_id = base.getid(share_group)
    body = {'share_group_id': share_group_id}
    if name:
        body['name'] = name
    if description:
        body['description'] = description
    return self._create(RESOURCES_PATH, {RESOURCE_NAME: body}, RESOURCE_NAME)