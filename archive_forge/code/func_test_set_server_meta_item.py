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
def test_set_server_meta_item(self):
    m = self.cs.servers.set_meta_item(1234, 'test_key', 'test_value')
    self.assert_request_id(m, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('PUT', '/servers/1234/metadata/test_key', {'meta': {'test_key': 'test_value'}})