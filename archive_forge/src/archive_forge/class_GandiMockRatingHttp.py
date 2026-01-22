import sys
import random
import string
import unittest
from libcloud.utils.py3 import httplib
from libcloud.common.gandi import GandiException
from libcloud.test.secrets import GANDI_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gandi import GandiNodeDriver
from libcloud.test.common.test_gandi import BaseGandiMockHttp
class GandiMockRatingHttp(BaseGandiMockHttp):
    """Fixtures needed for tests related to rating model"""
    fixtures = ComputeFileFixtures('gandi')

    def _xmlrpc__hosting_datacenter_list(self, method, url, body, headers):
        body = self.fixtures.load('datacenter_list.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__hosting_image_list(self, method, url, body, headers):
        body = self.fixtures.load('image_list_dc0.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__hosting_vm_create_from(self, method, url, body, headers):
        body = self.fixtures.load('vm_create_from.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__operation_info(self, method, url, body, headers):
        body = self.fixtures.load('operation_info.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__hosting_vm_info(self, method, url, body, headers):
        body = self.fixtures.load('vm_info.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__hosting_account_info(self, method, url, body, headers):
        body = self.fixtures.load('account_info_rating.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])