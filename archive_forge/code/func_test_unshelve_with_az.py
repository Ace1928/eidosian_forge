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
def test_unshelve_with_az(self):
    s = self.cs.servers.get(1234)
    ret = s.unshelve(availability_zone='foo-az')
    self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('POST', '/servers/1234/action', {'unshelve': {'availability_zone': 'foo-az'}})
    ret = self.cs.servers.unshelve(s, availability_zone='foo-az')
    self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('POST', '/servers/1234/action', {'unshelve': {'availability_zone': 'foo-az'}})