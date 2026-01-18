import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, Connection, XmlResponse, JsonResponse
from libcloud.test.file_fixtures import ComputeFileFixtures
def test_xml_unicode_response(self):
    self.connection.responseCls = XmlResponse
    response = self.connection.request('/unicode/xml')
    self.assertEqual(response.object.text, 'Ś')