from urllib import parse
from cinderclient.apiclient import base as common_base
from cinderclient import base
class VolumeType(base.Resource):
    """A Volume Type is the type of volume to be created."""

    def __repr__(self):
        return '<VolumeType: %s>' % self.name

    @property
    def is_public(self):
        """
        Provide a user-friendly accessor to os-volume-type-access:is_public
        """
        return self._info.get('os-volume-type-access:is_public', self._info.get('is_public', 'N/A'))

    def get_keys(self):
        """Get extra specs from a volume type.

        :param vol_type: The :class:`VolumeType` to get extra specs from
        """
        _resp, body = self.manager.api.client.get('/types/%s/extra_specs' % base.getid(self))
        return body['extra_specs']

    def set_keys(self, metadata):
        """Set extra specs on a volume type.

        :param type : The :class:`VolumeType` to set extra spec on
        :param metadata: A dict of key/value pairs to be set
        """
        body = {'extra_specs': metadata}
        return self.manager._create('/types/%s/extra_specs' % base.getid(self), body, 'extra_specs', return_raw=True)

    def unset_keys(self, keys):
        """Unset extra specs on a volue type.

        :param type_id: The :class:`VolumeType` to unset extra spec on
        :param keys: A list of keys to be unset
        """
        response_list = []
        for k in keys:
            resp, body = self.manager._delete('/types/%s/extra_specs/%s' % (base.getid(self), k))
            response_list.append(resp)
        return common_base.ListWithMeta([], response_list)