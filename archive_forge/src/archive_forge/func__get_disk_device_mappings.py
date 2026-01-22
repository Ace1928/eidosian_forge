import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _get_disk_device_mappings(self, element):
    if element is None:
        return None
    mapping_element = element.find('DiskDeviceMapping')
    if mapping_element is not None:
        return self._get_extra_dict(mapping_element, RESOURCE_EXTRA_ATTRIBUTES_MAP['disk_device_mapping'])
    return None