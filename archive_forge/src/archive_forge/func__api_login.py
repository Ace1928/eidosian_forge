import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def _api_login(self, method, url, body, headers):
    if headers['Authorization'] == 'Basic c29uOmdvdGVu':
        expected_response = self.fixtures.load('unauthorized_user.html')
        expected_status = httplib.UNAUTHORIZED
    else:
        expected_response = self.fixtures.load('login.xml')
        expected_status = httplib.OK
    return (expected_status, expected_response, {}, '')