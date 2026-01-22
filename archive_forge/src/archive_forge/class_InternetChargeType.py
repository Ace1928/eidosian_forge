import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
class InternetChargeType:
    """
    Internet connection billing types for Aliyun Nodes.
    """
    BY_BANDWIDTH = 'PayByBandwidth'
    BY_TRAFFIC = 'PayByTraffic'