import typing as ty
from openstack.block_storage import _base_proxy
from openstack.block_storage.v3 import attachment as _attachment
from openstack.block_storage.v3 import availability_zone
from openstack.block_storage.v3 import backup as _backup
from openstack.block_storage.v3 import block_storage_summary as _summary
from openstack.block_storage.v3 import capabilities as _capabilities
from openstack.block_storage.v3 import extension as _extension
from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.block_storage.v3 import group_type as _group_type
from openstack.block_storage.v3 import limits as _limits
from openstack.block_storage.v3 import quota_set as _quota_set
from openstack.block_storage.v3 import resource_filter as _resource_filter
from openstack.block_storage.v3 import service as _service
from openstack.block_storage.v3 import snapshot as _snapshot
from openstack.block_storage.v3 import stats as _stats
from openstack.block_storage.v3 import transfer as _transfer
from openstack.block_storage.v3 import type as _type
from openstack.block_storage.v3 import volume as _volume
from openstack import exceptions
from openstack.identity.v3 import project as _project
from openstack import resource
from openstack import utils
def upload_volume_to_image(self, volume, image_name, force=False, disk_format=None, container_format=None, visibility=None, protected=None):
    """Uploads the specified volume to image service.

        :param volume: The value can be either the ID of a volume or a
            :class:`~openstack.block_storage.v3.volume.Volume` instance.
        :param str image name: The name for the new image.
        :param bool force: Enables or disables upload of a volume that is
            attached to an instance.
        :param str disk_format: Disk format for the new image.
        :param str container_format: Container format for the new image.
        :param str visibility: The visibility property of the new image.
        :param str protected: Whether the new image is protected.

        :returns: dictionary describing the image.
        """
    volume = self._get_resource(_volume.Volume, volume)
    return volume.upload_to_image(self, image_name, force=force, disk_format=disk_format, container_format=container_format, visibility=visibility, protected=protected)