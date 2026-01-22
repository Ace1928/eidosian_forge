import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CLOUDSCALE_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudscale import CloudscaleNodeDriver
class CloudscaleMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('cloudscale')

    def _v1_images(self, method, url, body, headers):
        body = self.fixtures.load('list_images.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_flavors(self, method, url, body, headers):
        body = self.fixtures.load('list_sizes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_servers(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_nodes.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            body = self.fixtures.load('create_node.json')
            response = httplib.responses[httplib.CREATED]
            return (httplib.CREATED, body, {}, response)

    def _v1_servers_47cec963_fcd2_482f_bdb6_24461b2d47b1(self, method, url, body, headers):
        assert method == 'DELETE'
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])

    def _v1_servers_47cec963_fcd2_482f_bdb6_24461b2d47b1_reboot(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])