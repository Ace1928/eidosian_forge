import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import ServiceUnavailableError
from libcloud.compute.base import NodeSize, NodeImage
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV1
def _v1_server_list_SERVICE_UNAVAILABLE(self, method, url, body, headers):
    body = self.fixtures.load('error_rate_limit.txt')
    return (httplib.SERVICE_UNAVAILABLE, body, {}, httplib.responses[httplib.SERVICE_UNAVAILABLE])