import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
class CloudSigmaHttp(MockHttp):
    fixtures = ComputeFileFixtures('cloudsigma')

    def _drives_standard_info(self, method, url, body, headers):
        body = self.fixtures.load('drives_standard_info.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _servers_62fe7cde_4fb9_4c63_bd8c_e757930066a0_start(self, method, url, body, headers):
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _servers_62fe7cde_4fb9_4c63_bd8c_e757930066a0_stop(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.OK])

    def _servers_62fe7cde_4fb9_4c63_bd8c_e757930066a0_destroy(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _drives_d18119ce_7afa_474a_9242_e0384b160220_clone(self, method, url, body, headers):
        body = self.fixtures.load('drives_clone.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _drives_a814def5_1789_49a0_bf88_7abe7bb1682a_info(self, method, url, body, headers):
        body = self.fixtures.load('drives_single_info.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _drives_info(self, method, url, body, headers):
        body = self.fixtures.load('drives_info.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _servers_create(self, method, url, body, headers):
        body = self.fixtures.load('servers_create.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _servers_info(self, method, url, body, headers):
        body = self.fixtures.load('servers_info.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _resources_ip_list(self, method, url, body, headers):
        body = self.fixtures.load('resources_ip_list.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _resources_ip_create(self, method, url, body, headers):
        body = self.fixtures.load('resources_ip_create.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _resources_ip_1_2_3_4_destroy(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.OK])

    def _drives_d18119ce_7afa_474a_9242_e0384b160220_destroy(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.OK])

    def _servers_62fe7cde_4fb9_4c63_bd8c_e757930066a0_set(self, method, url, body, headers):
        body = self.fixtures.load('servers_set.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])