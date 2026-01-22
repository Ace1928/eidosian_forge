import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_3_2_MockHttp(OpenNebula_3_0_MockHttp):
    """
    Mock HTTP server for testing v3.2 of the OpenNebula.org compute driver.
    """
    fixtures_3_2 = ComputeFileFixtures('opennebula_3_2')

    def _compute_5(self, method, url, body, headers):
        """
        Compute entry resource.
        """
        if method == 'GET':
            body = self.fixtures.load('compute_5.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'PUT':
            body = ''
            return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _instance_type(self, method, url, body, headers):
        """
        Instance type pool.
        """
        if method == 'GET':
            body = self.fixtures_3_2.load('instance_type_collection.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])