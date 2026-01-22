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
def _v3_OS_FEDERATION_identity_providers_idp_protocols_oidc_auth(self, method, url, body, headers):
    if method == 'GET':
        headers = self.json_content_headers.copy()
        headers['x-subject-token'] = '00000000000000000000000000000000'
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])
    raise NotImplementedError()