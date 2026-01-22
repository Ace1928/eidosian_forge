import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_bare_metals_234_start(self, method, url, body, headers):
    return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])