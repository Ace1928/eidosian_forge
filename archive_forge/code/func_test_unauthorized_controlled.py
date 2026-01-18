import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_unauthorized_controlled(self):
    """
        Test the Unauthorized Exception is Controlled.

        Test, through the 'login' method, that a '401 Unauthorized'
        raises a 'InvalidCredsError' instead of the 'MalformedUrlException'
        """
    self.assertRaises(InvalidCredsError, AbiquoNodeDriver, 'son', 'goten', 'http://dummy.host.com/api')