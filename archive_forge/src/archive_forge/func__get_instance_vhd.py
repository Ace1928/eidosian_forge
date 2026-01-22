import os
import time
import base64
import binascii
from libcloud.utils import iso8601
from libcloud.utils.py3 import parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.storage.types import ObjectDoesNotExistError
from libcloud.common.azure_arm import AzureResourceManagementConnection
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import Provider
from libcloud.storage.drivers.azure_blobs import AzureBlobsStorageDriver
def _get_instance_vhd(self, name, ex_resource_group, ex_storage_account, ex_blob_container='vhds'):
    n = 0
    errors = []
    while n < 10:
        try:
            instance_vhd = 'https://%s.blob%s/%s/%s-os_%i.vhd' % (ex_storage_account, self.connection.storage_suffix, ex_blob_container, name, n)
            if self._ex_delete_old_vhd(ex_resource_group, instance_vhd):
                return instance_vhd
        except LibcloudError as lce:
            errors.append(str(lce))
        n += 1
    raise LibcloudError('Unable to find a name for a VHD to use for instance in 10 tries, errors were:\n  - %s' % '\n  - '.join(errors))