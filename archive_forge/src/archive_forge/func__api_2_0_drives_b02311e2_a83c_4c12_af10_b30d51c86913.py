import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_drives_b02311e2_a83c_4c12_af10_b30d51c86913(self, method, url, body, headers):
    body = self.fixtures.load('drives_get.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])