import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_3_6_MockHttp(OpenNebula_3_2_MockHttp):
    """
    Mock HTTP server for testing v3.6 of the OpenNebula.org compute driver.
    """
    fixtures_3_6 = ComputeFileFixtures('opennebula_3_6')

    def _storage(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('storage_collection.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'POST':
            body = self.fixtures_3_6.load('storage_5.xml')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _compute_5(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures_3_6.load('compute_5.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'PUT':
            body = ''
            return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _compute_5_action(self, method, url, body, headers):
        body = self.fixtures_3_6.load('compute_5.xml')
        if method == 'POST':
            return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
        if method == 'GET':
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _compute_15(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures_3_6.load('compute_15.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'PUT':
            body = ''
            return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _storage_10(self, method, url, body, headers):
        """
        Storage entry resource.
        """
        if method == 'GET':
            body = self.fixtures_3_6.load('disk_10.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _storage_15(self, method, url, body, headers):
        """
        Storage entry resource.
        """
        if method == 'GET':
            body = self.fixtures_3_6.load('disk_15.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])