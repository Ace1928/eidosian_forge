import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_instances_123(self, method, url, body, headers):
    if method == 'DELETE':
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])
    elif method == 'GET':
        body = self.fixtures.load('ex_get_node.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'PATCH':
        body = self.fixtures.load('ex_resize_node.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])