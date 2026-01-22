import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_ssh_keys_123(self, method, url, body, headers):
    if method == 'DELETE':
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'GET':
        body = self.fixtures.load('get_key_pair.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])