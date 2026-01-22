import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import KAMATERA_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kamatera import KamateraNodeDriver
class KamateraAuthenticationTests(LibcloudTestCase):

    def setUp(self):
        KamateraNodeDriver.connectionCls.conn_class = KamateraMockHttp
        self.driver = KamateraNodeDriver('nosuchuser', 'nopwd')

    def test_authentication_fails(self):
        with self.assertRaises(BaseHTTPError):
            self.driver.list_locations()