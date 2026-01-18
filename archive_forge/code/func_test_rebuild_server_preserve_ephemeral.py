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
def test_rebuild_server_preserve_ephemeral(self):
    s = self.cs.servers.get(1234)
    ret = s.rebuild(image=1, preserve_ephemeral=True)
    self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('POST', '/servers/1234/action')
    d = self.requests_mock.last_request.json()['rebuild']
    self.assertIn('preserve_ephemeral', d)
    self.assertTrue(d['preserve_ephemeral'])