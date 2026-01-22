from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _create_share_group(self, share_group_type=None, share_types=None, share_network=None, name=None, description=None, source_share_group_snapshot=None, availability_zone=None):
    """Create a Share Group.

        :param share_group_type: either instance of ShareGroupType or text
            with UUID
        :param share_types: list of the share types allowed in the group. May
            not be supplied when 'source_group_snapshot_id' is provided.  These
            may be ShareType objects or UUIDs.
        :param share_network: either the share network object or text of the
            UUID - represents the share network to use when creating a
            share group when driver_handles_share_servers = True.
        :param name: text - name of the new share group
        :param description: text - description of the share group
        :param source_share_group_snapshot: text - either instance of
            ShareGroupSnapshot or text with UUID from which this shar_group is
            to be created. May not be supplied when 'share_types' is provided.
        :param availability_zone: name of the availability zone where the
            group is to be created
        :rtype: :class:`ShareGroup`
        """
    if share_types and source_share_group_snapshot:
        raise ValueError('Cannot specify a share group with bothshare_types and source_share_group_snapshot.')
    body = {}
    if name:
        body['name'] = name
    if description:
        body['description'] = description
    if availability_zone:
        body['availability_zone'] = availability_zone
    if share_group_type:
        body['share_group_type_id'] = base.getid(share_group_type)
    if share_network:
        body['share_network_id'] = base.getid(share_network)
    if source_share_group_snapshot:
        body['source_share_group_snapshot_id'] = base.getid(source_share_group_snapshot)
    elif share_types:
        body['share_types'] = [base.getid(share_type) for share_type in share_types]
    return self._create(RESOURCES_PATH, {RESOURCE_NAME: body}, RESOURCE_NAME)