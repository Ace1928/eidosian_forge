import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _get_system_disk(self, ex_system_disk):
    if not isinstance(ex_system_disk, dict):
        raise AttributeError('ex_system_disk is not a dict')
    sys_disk_dict = ex_system_disk
    key_base = 'SystemDisk.'
    mappings = {'category': 'Category', 'disk_name': 'DiskName', 'description': 'Description'}
    params = {}
    for attr in mappings.keys():
        if attr in sys_disk_dict:
            params[key_base + mappings[attr]] = sys_disk_dict[attr]
    return params