import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, Connection, XmlResponse, JsonResponse
from libcloud.test.file_fixtures import ComputeFileFixtures
def _unicode(self, method, url, body, headers):
    body = self.fixtures.load('unicode.txt')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])