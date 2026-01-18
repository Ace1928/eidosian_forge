from libcloud.utils.py3 import httplib
from libcloud.common.ovh import API_ROOT, OvhConnection
from libcloud.compute.base import (
from libcloud.compute.types import Provider, StorageVolumeState, VolumeSnapshotState
from libcloud.compute.drivers.openstack import OpenStackKeyPair, OpenStackNodeDriver

        Create snapshot from volume

        :param volume: Instance of `StorageVolume`
        :type  volume: `StorageVolume`

        :param name: Name of snapshot (optional)
        :type  name: `str` | `NoneType`

        :param ex_description: Description of the snapshot (optional)
        :type  ex_description: `str` | `NoneType`

        :rtype: :class:`VolumeSnapshot`
        