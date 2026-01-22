import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_tags_a010ec41_2ead_4630_a1d0_237fa77e4d4d(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('tags_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'PUT':
        body = self.fixtures.load('tags_update.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'DELETE':
        body = ''
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])