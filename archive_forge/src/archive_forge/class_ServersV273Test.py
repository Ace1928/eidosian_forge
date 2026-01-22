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
class ServersV273Test(ServersV268Test):
    api_version = '2.73'

    def test_lock_server(self):
        s = self.cs.servers.get(1234)
        ret = s.lock()
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'lock': None})
        ret = s.lock(reason='zombie-apocalypse')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'lock': {'locked_reason': 'zombie-apocalypse'}})

    def test_lock_server_pre_273_fails_with_reason(self):
        self.cs.api_version = api_versions.APIVersion('2.72')
        s = self.cs.servers.get(1234)
        e = self.assertRaises(TypeError, s.lock, reason='blah')
        self.assertIn("unexpected keyword argument 'reason'", str(e))

    def test_filter_servers_unlocked(self):
        sl = self.cs.servers.list(search_opts={'locked': False})
        self.assert_request_id(sl, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('GET', '/servers/detail?locked=False')
        for s in sl:
            self.assertIsInstance(s, servers.Server)