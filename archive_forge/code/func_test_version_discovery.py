import argparse
import io
import json
import re
import sys
from unittest import mock
import ddt
import fixtures
import keystoneauth1.exceptions as ks_exc
from keystoneauth1.exceptions import DiscoveryFailure
from keystoneauth1.identity.generic.password import Password as ks_password
from keystoneauth1 import session
import requests_mock
from testtools import matchers
import cinderclient
from cinderclient import api_versions
from cinderclient.contrib import noauth
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit import fake_actions_module
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@requests_mock.Mocker()
def test_version_discovery(self, mocker):
    _shell = shell.OpenStackCinderShell()
    sess = session.Session()
    os_auth_url = 'https://wrongdiscoveryresponse.discovery.com:35357/v2.0'
    self.register_keystone_auth_fixture(mocker, os_auth_url)
    self.assertRaises(DiscoveryFailure, _shell._discover_auth_versions, sess, auth_url=os_auth_url)
    os_auth_url = 'https://DiscoveryNotSupported.discovery.com:35357/v2.0'
    self.register_keystone_auth_fixture(mocker, os_auth_url)
    v2_url, v3_url = _shell._discover_auth_versions(sess, auth_url=os_auth_url)
    self.assertEqual(os_auth_url, v2_url, 'Expected v2 url')
    self.assertIsNone(v3_url, 'Expected no v3 url')
    os_auth_url = 'https://DiscoveryNotSupported.discovery.com:35357/v3.0'
    self.register_keystone_auth_fixture(mocker, os_auth_url)
    v2_url, v3_url = _shell._discover_auth_versions(sess, auth_url=os_auth_url)
    self.assertEqual(os_auth_url, v3_url, 'Expected v3 url')
    self.assertIsNone(v2_url, 'Expected no v2 url')