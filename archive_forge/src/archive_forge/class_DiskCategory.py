import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
class DiskCategory:
    """
    Enum defined disk types supported by Aliyun system and data disks.
    """
    CLOUD = 'cloud'
    CLOUD_EFFICIENCY = 'cloud_efficiency'
    CLOUD_SSD = 'cloud_ssd'
    EPHEMERAL_SSD = 'ephemeral_ssd'