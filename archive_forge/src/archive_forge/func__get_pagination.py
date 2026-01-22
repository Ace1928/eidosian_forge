import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _get_pagination(self, element):
    page_number = int(findtext(element, 'PageNumber'))
    total_count = int(findtext(element, 'TotalCount'))
    page_size = int(findtext(element, 'PageSize'))
    return Pagination(total=total_count, size=page_size, current=page_number)