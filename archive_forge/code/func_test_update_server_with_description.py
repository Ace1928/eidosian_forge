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
def test_update_server_with_description(self):
    s = self.cs.servers.get(1234)
    s.update(description='hi')
    s.update(name='hi', description='hi')
    self.assert_called('PUT', '/servers/1234')