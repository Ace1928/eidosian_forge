import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def _create_volume_from_template(self, size, name, location=None, template=None):
    """
        create Storage

        :param name: name of your Storage unit
        :type name: ``str``

        :param size: Integer in GB.
        :type size: ``int``

        :param location: your server location
        :type location: :class:`.NodeLocation`

        :param template: template to shape the storage capacity to
        :type template: ``dict``

        :return: newly created StorageVolume
        :rtype: :class:`.GridscaleVolumeStorage`
        """
    template = template
    self.connection.async_request('objects/storages/', data={'name': name, 'capacity': size, 'location_uuid': location.id, 'template': template}, method='POST')
    return self._to_volume(self._get_resource('storages', self.connection.poll_response_initial.object['object_uuid']))