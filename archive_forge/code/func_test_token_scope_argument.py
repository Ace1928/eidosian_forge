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
def test_token_scope_argument(self):
    expected_msg = 'Invalid value for "token_scope" argument: foo'
    assertRaisesRegex(self, ValueError, expected_msg, OpenStackIdentity_3_0_Connection, auth_url='http://none', user_id='test', key='test', token_scope='foo')
    expected_msg = 'Must provide tenant_name and domain_name argument'
    assertRaisesRegex(self, ValueError, expected_msg, OpenStackIdentity_3_0_Connection, auth_url='http://none', user_id='test', key='test', token_scope='project')
    expected_msg = 'Must provide domain_name argument'
    assertRaisesRegex(self, ValueError, expected_msg, OpenStackIdentity_3_0_Connection, auth_url='http://none', user_id='test', key='test', token_scope='domain', domain_name=None)
    OpenStackIdentity_3_0_Connection(auth_url='http://none', user_id='test', key='test', token_scope='project', tenant_name='test', domain_name='Default')
    OpenStackIdentity_3_0_Connection(auth_url='http://none', user_id='test', key='test', token_scope='domain', tenant_name=None, domain_name='Default')