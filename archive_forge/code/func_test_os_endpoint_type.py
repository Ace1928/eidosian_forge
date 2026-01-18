import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
@requests_mock.Mocker()
def test_os_endpoint_type(self, m_requests):
    self.make_env(exclude='NOVA_ENDPOINT_TYPE', fake_env=FAKE_ENV3)
    self.register_keystone_discovery_fixture(m_requests)
    self.shell('list')
    client_kwargs = self.mock_client.call_args_list[0][1]
    self.assertEqual(client_kwargs['endpoint_type'], 'osURL')