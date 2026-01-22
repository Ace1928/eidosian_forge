import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_drives_availability_groups(self, method, url, body, headers):
    body = self.fixtures.load('drives_avail_groups.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])