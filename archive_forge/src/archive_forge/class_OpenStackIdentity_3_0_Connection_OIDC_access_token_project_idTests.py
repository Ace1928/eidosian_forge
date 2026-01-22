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
class OpenStackIdentity_3_0_Connection_OIDC_access_token_project_idTests(unittest.TestCase):

    def setUp(self):
        mock_cls = OpenStackIdentity_3_0_MockHttp
        mock_cls.type = None
        OpenStackIdentity_3_0_Connection_OIDC_access_token.conn_class = mock_cls
        self.auth_instance = OpenStackIdentity_3_0_Connection_OIDC_access_token(auth_url='http://none', user_id='idp', key='token', tenant_name='oidc', domain_name='project_id2')
        self.auth_instance.auth_token = 'mock'

    def test_authenticate_valid_project_id(self):
        auth = OpenStackIdentity_3_0_Connection_OIDC_access_token(auth_url='http://none', user_id='idp', key='token', token_scope='project', tenant_name='oidc', domain_name='project_id2')
        auth.authenticate()

    def test_authenticate_invalid_project_id(self):
        auth = OpenStackIdentity_3_0_Connection_OIDC_access_token(auth_url='http://none', user_id='idp', key='token', token_scope='project', tenant_name='oidc', domain_name='project_id100')
        expected_msg = 'Project project_id100 not found'
        self.assertRaisesRegex(ValueError, expected_msg, auth.authenticate)