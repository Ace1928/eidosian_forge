import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_blocks(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('list_volumes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'POST':
        body = self.fixtures.load('create_volume.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])