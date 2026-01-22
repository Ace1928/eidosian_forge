import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
class OpenStackIdentity_3_0_Connection_AppCredTests(unittest.TestCase):

    def setUp(self):
        mock_cls = OpenStackIdentity_3_0_AppCred_MockHttp
        mock_cls.type = None
        OpenStackIdentity_3_0_Connection_AppCred.conn_class = mock_cls
        self.auth_instance = OpenStackIdentity_3_0_Connection_AppCred(auth_url='http://none', user_id='appcred_id', key='appcred_secret', proxy_url='http://proxy:8080', timeout=10)
        self.auth_instance.auth_token = 'mock'

    def test_authenticate(self):
        auth = OpenStackIdentity_3_0_Connection_AppCred(auth_url='http://none', user_id='appcred_id', key='appcred_secret', proxy_url='http://proxy:8080', timeout=10)
        auth.authenticate()