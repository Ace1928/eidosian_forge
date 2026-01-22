import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, Connection, XmlResponse, JsonResponse
from libcloud.test.file_fixtures import ComputeFileFixtures
class MockHttpFileFixturesTests(unittest.TestCase):
    """
    Test the behaviour of MockHttp
    """

    def setUp(self):
        Connection.conn_class = TestMockHttp
        Connection.responseCls = Response
        self.connection = Connection()

    def test_unicode_response(self):
        r = self.connection.request('/unicode')
        self.assertEqual(r.parse_body(), 'Ś')

    def test_json_unicode_response(self):
        self.connection.responseCls = JsonResponse
        r = self.connection.request('/unicode/json')
        self.assertEqual(r.object, {'test': 'Ś'})

    def test_xml_unicode_response(self):
        self.connection.responseCls = XmlResponse
        response = self.connection.request('/unicode/xml')
        self.assertEqual(response.object.text, 'Ś')