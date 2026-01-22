import sys
import base64
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import BRIGHTBOX_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.brightbox import BrightboxNodeDriver
def _1_0_cloud_ips(self, method, url, body, headers):
    if method == 'GET':
        return self.test_response(httplib.OK, self.fixtures.load('list_cloud_ips.json'))
    elif method == 'POST':
        if body:
            body = json.loads(body)
        node = json.loads(self.fixtures.load('create_cloud_ip.json'))
        if 'reverse_dns' in body:
            node['reverse_dns'] = body['reverse_dns']
        return self.test_response(httplib.ACCEPTED, json.dumps(node))