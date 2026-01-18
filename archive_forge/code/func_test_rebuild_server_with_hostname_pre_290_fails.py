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
def test_rebuild_server_with_hostname_pre_290_fails(self):
    self.cs.api_version = api_versions.APIVersion('2.89')
    ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.rebuild, '1234', fakes.FAKE_IMAGE_UUID_1, hostname='new-hostname')
    self.assertIn('hostname', str(ex))