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
class ServersV219Test(ServersV217Test):
    api_version = '2.19'

    def test_create_server_with_description(self):
        self.cs.servers.create(name='My server', description='descr', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', nics=self._get_server_create_default_nics())
        self.assert_called('POST', '/servers')

    def test_update_server_with_description(self):
        s = self.cs.servers.get(1234)
        s.update(description='hi')
        s.update(name='hi', description='hi')
        self.assert_called('PUT', '/servers/1234')

    def test_rebuild_with_description(self):
        s = self.cs.servers.get(1234)
        ret = s.rebuild(image='1', description='descr')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action')