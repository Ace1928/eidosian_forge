import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_os(self, method, url, body, headers):
    body = self.fixtures.load('list_images.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])