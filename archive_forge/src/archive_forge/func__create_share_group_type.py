from manilaclient import api_versions
from manilaclient import base
def _create_share_group_type(self, name, share_types, is_public=False, group_specs=None):
    """Create a share group type.

        :param name: Descriptive name of the share group type
        :param share_types: list of either instances of ShareType or text
           with share type UUIDs
        :param is_public: True to create a public share group type
        :param group_specs: dict containing group spec key-value pairs
        :rtype: :class:`ShareGroupType`
        """
    if not share_types:
        raise ValueError('At least one share type must be specified when creating a share group type.')
    body = {'name': name, 'is_public': is_public, 'group_specs': group_specs or {}, 'share_types': [base.getid(share_type) for share_type in share_types]}
    return self._create(RESOURCES_PATH, {RESOURCE_NAME: body}, RESOURCE_NAME)