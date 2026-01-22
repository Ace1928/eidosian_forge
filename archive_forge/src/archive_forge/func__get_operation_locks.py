import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _get_operation_locks(self, instance):
    locks = findall(instance, xpath='OperationLocks', namespace=self.namespace)
    if len(locks) <= 0:
        return None
    return self._get_extra_dict(locks[0], RESOURCE_EXTRA_ATTRIBUTES_MAP['operation_locks'])