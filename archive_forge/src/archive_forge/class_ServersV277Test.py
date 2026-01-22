import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
class ServersV277Test(ServersV274Test):
    api_version = '2.77'

    def test_unshelve(self):
        s = self.cs.servers.get(1234)
        ret = s.unshelve()
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': None})
        ret = self.cs.servers.unshelve(s)
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': None})

    def test_unshelve_with_az(self):
        s = self.cs.servers.get(1234)
        ret = s.unshelve(availability_zone='foo-az')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'availability_zone': 'foo-az'}})
        ret = self.cs.servers.unshelve(s, availability_zone='foo-az')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'unshelve': {'availability_zone': 'foo-az'}})

    def test_unshelve_server_pre_277_fails_with_specified_az(self):
        self.cs.api_version = api_versions.APIVersion('2.76')
        s = self.cs.servers.get(1234)
        ex = self.assertRaises(TypeError, s.unshelve, availability_zone='foo-az')
        self.assertIn("unexpected keyword argument 'availability_zone'", str(ex))
        ex = self.assertRaises(TypeError, self.cs.servers.unshelve, s, availability_zone='foo-az')
        self.assertIn("unexpected keyword argument 'availability_zone'", str(ex))