import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, Connection, XmlResponse, JsonResponse
from libcloud.test.file_fixtures import ComputeFileFixtures
def test_unicode_response(self):
    r = self.connection.request('/unicode')
    self.assertEqual(r.parse_body(), 'Ś')