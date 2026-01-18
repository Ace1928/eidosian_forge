from manilaclient import api_versions
from manilaclient import base
@api_versions.wraps('2.45')
def unset_metadata(self, access, keys):
    """Unset metadata on a share access rule.

        :param keys: A list of keys on this object to be unset
        :return: None if successful, else API response on failure
        """
    for k in keys:
        url = RESOURCE_METADATA_PATH % (base.getid(access), k)
        self._delete(url)