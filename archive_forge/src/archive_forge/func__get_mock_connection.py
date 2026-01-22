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
def _get_mock_connection(self, mock_http_class, auth_url=None):
    OpenStackBaseConnection.conn_class = mock_http_class
    if auth_url is None:
        auth_url = 'https://auth.api.example.com'
    OpenStackBaseConnection.auth_url = auth_url
    connection = OpenStackBaseConnection(*OPENSTACK_PARAMS)
    connection._ex_force_base_url = 'https://www.foo.com'
    connection.driver = OpenStack_1_0_NodeDriver(*OPENSTACK_PARAMS)
    return connection