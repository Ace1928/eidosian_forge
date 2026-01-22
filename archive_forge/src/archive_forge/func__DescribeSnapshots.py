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
def _DescribeSnapshots(self, method, url, body, headers):
    region = self.test.region
    volume_id = self.test.fake_volume.id
    params = {'RegionId': region, 'DiskId': volume_id}
    self.assertUrlContainsQueryParams(url, params)
    resp_body = self.fixtures.load('describe_snapshots.xml')
    return (httplib.OK, resp_body, {}, httplib.responses[httplib.OK])