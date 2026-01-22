import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
class ECSZone:
    """
    ECSZone used to represent an availability zone in a region.
    """

    def __init__(self, id, name, driver=None, available_resource_types=None, available_instance_types=None, available_disk_categories=None):
        self.id = id
        self.name = name
        self.driver = driver
        self.available_resource_types = available_resource_types
        self.available_instance_types = available_instance_types
        self.available_disk_categories = available_disk_categories

    def __repr__(self):
        return '<ECSZone: id={}, name={}, driver={}>'.format(self.id, self.name, self.driver)