import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_close_vnc(self, method, url, body, headers):
    body = self.fixtures.load('servers_close_vnc.json')
    return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])