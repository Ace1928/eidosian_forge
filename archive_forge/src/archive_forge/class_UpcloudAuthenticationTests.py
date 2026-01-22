import re
import sys
import json
import base64
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib, ensure_string
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import UPCLOUD_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.upcloud import UpcloudDriver, UpcloudResponse
class UpcloudAuthenticationTests(LibcloudTestCase):

    def setUp(self):
        UpcloudDriver.connectionCls.conn_class = UpcloudMockHttp
        self.driver = UpcloudDriver('nosuchuser', 'nopwd')

    def test_authentication_fails(self):
        with self.assertRaises(InvalidCredsError):
            self.driver.list_locations()