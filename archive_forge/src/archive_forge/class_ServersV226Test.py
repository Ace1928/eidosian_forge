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
class ServersV226Test(ServersV225Test):
    api_version = '2.26'

    def test_tag_list(self):
        s = self.cs.servers.get(1234)
        ret = s.tag_list()
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('GET', '/servers/1234/tags')

    def test_tag_delete(self):
        s = self.cs.servers.get(1234)
        ret = s.delete_tag('tag')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('DELETE', '/servers/1234/tags/tag')

    def test_tag_delete_all(self):
        s = self.cs.servers.get(1234)
        ret = s.delete_all_tags()
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('DELETE', '/servers/1234/tags')

    def test_tag_add(self):
        s = self.cs.servers.get(1234)
        ret = s.add_tag('tag')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('PUT', '/servers/1234/tags/tag')

    def test_tags_set(self):
        s = self.cs.servers.get(1234)
        ret = s.set_tags(['tag1', 'tag2'])
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('PUT', '/servers/1234/tags')