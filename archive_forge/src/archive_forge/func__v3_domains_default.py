import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
def _v3_domains_default(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('v3_domains_default.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
    raise NotImplementedError()