import os
import sys
import datetime
import unittest
from unittest import mock
from unittest.mock import Mock, patch
import pytest
import requests_mock
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.pricing import set_pricing, clear_pricing_data
from libcloud.utils.py3 import u, httplib, method_type
from libcloud.common.base import LibcloudConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import OpenStackFixtures, ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import (
@patch('libcloud.common.openstack.OpenStackServiceCatalog')
def test_populate_hosts_and_requests_path(self, _):
    tomorrow = datetime.datetime.today() + datetime.timedelta(1)
    cls = self.driver_klass.connectionCls
    count = 5
    con = cls('username', 'key')
    osa = con.get_auth_class()
    mocked_auth_method = Mock()
    osa.authenticate = mocked_auth_method
    for i in range(0, count):
        con._populate_hosts_and_request_paths()
        if i == 0:
            osa.auth_token = '1234'
            osa.auth_token_expires = tomorrow
    self.assertEqual(mocked_auth_method.call_count, 1)
    osa.auth_token = None
    osa.auth_token_expires = None
    con = cls('username', 'key', ex_force_base_url='http://ponies', ex_force_auth_token='1234')
    osa = con.get_auth_class()
    mocked_auth_method = Mock()
    osa.authenticate = mocked_auth_method
    for i in range(0, count):
        con._populate_hosts_and_request_paths()
    self.assertEqual(mocked_auth_method.call_count, 0)