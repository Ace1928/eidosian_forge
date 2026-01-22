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
class OpenStackIdentity_2_0_Connection_VOMSMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('openstack_identity/v2')
    json_content_headers = {'content-type': 'application/json; charset=UTF-8'}

    def _v2_0_tokens(self, method, url, body, headers):
        if method == 'POST':
            status = httplib.UNAUTHORIZED
            data = json.loads(body)
            if 'voms' in data['auth'] and data['auth']['voms'] is True:
                status = httplib.OK
            body = ComputeFileFixtures('openstack').load('_v2_0__auth.json')
            headers = self.json_content_headers.copy()
            headers['x-subject-token'] = '00000000000000000000000000000000'
            return (status, body, headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v2_0_tenants(self, method, url, body, headers):
        if method == 'GET':
            body = json.dumps({'tenant': [{'name': 'tenant_name'}]})
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()