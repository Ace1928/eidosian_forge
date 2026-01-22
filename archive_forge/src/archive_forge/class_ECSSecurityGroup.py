import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
class ECSSecurityGroup:
    """
    Security group used to control nodes internet and intranet accessibility.
    """

    def __init__(self, id, name, description=None, driver=None, vpc_id=None, creation_time=None):
        self.id = id
        self.name = name
        self.description = description
        self.driver = driver
        self.vpc_id = vpc_id
        self.creation_time = creation_time

    def __repr__(self):
        return '<ECSSecurityGroup: id={}, name={}, driver={} ...>'.format(self.id, self.name, self.driver.name)