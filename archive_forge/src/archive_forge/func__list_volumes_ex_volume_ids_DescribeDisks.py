import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.test.secrets import ECS_PARAMS
from libcloud.compute.types import NodeState, StorageVolumeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ecs import ECSDriver
def _list_volumes_ex_volume_ids_DescribeDisks(self, method, url, body, headers):
    region = self.test.region
    params = {'DiskIds': '["i-28n7dkvov", "not-existed-id"]', 'RegionId': region}
    self.assertUrlContainsQueryParams(url, params)
    return self._DescribeInstances(method, url, body, headers)