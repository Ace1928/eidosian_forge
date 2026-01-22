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
class ServersV230Test(ServersV229Test):
    api_version = '2.30'

    def test_live_migrate_server(self):
        s = self.cs.servers.get(1234)
        ret = s.live_migrate(host='hostname', block_migration='auto')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'os-migrateLive': {'host': 'hostname', 'block_migration': 'auto'}})
        ret = self.cs.servers.live_migrate(s, host='hostname', block_migration='auto', force=True)
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'os-migrateLive': {'host': 'hostname', 'block_migration': 'auto', 'force': True}})