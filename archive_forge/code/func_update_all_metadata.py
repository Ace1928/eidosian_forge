from cinderclient.apiclient import base as common_base
from cinderclient import base
def update_all_metadata(self, volume, metadata):
    """Update all metadata of a volume.

        :param volume: The :class:`Volume`.
        :param metadata: A list of keys to be updated.
        """
    body = {'metadata': metadata}
    return self._update('/volumes/%s/metadata' % base.getid(volume), body)