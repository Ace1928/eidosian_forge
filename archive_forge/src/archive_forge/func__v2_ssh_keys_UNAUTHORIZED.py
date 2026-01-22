import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_ssh_keys_UNAUTHORIZED(self, method, url, body, headers):
    body = '{"error":"Invalid API token.","status":401}'
    return (httplib.UNAUTHORIZED, body, {}, httplib.responses[httplib.UNAUTHORIZED])