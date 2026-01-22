import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CLOUDSCALE_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudscale import CloudscaleNodeDriver
def _v1_servers(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('list_nodes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    else:
        body = self.fixtures.load('create_node.json')
        response = httplib.responses[httplib.CREATED]
        return (httplib.CREATED, body, {}, response)