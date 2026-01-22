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
class ServersV26Test(ServersTest):
    api_version = '2.6'

    def test_get_vnc_console(self):
        s = self.cs.servers.get(1234)
        vc = s.get_vnc_console('novnc')
        self.assert_request_id(vc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        vc = self.cs.servers.get_vnc_console(s, 'novnc')
        self.assert_request_id(vc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        self.assertRaises(exceptions.UnsupportedConsoleType, s.get_vnc_console, 'invalid')

    def test_get_spice_console(self):
        s = self.cs.servers.get(1234)
        sc = s.get_spice_console('spice-html5')
        self.assert_request_id(sc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        sc = self.cs.servers.get_spice_console(s, 'spice-html5')
        self.assert_request_id(sc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        self.assertRaises(exceptions.UnsupportedConsoleType, s.get_spice_console, 'invalid')

    def test_get_serial_console(self):
        s = self.cs.servers.get(1234)
        sc = s.get_serial_console('serial')
        self.assert_request_id(sc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        sc = self.cs.servers.get_serial_console(s, 'serial')
        self.assert_request_id(sc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        self.assertRaises(exceptions.UnsupportedConsoleType, s.get_serial_console, 'invalid')

    def test_get_rdp_console(self):
        s = self.cs.servers.get(1234)
        rc = s.get_rdp_console('rdp-html5')
        self.assert_request_id(rc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        rc = self.cs.servers.get_rdp_console(s, 'rdp-html5')
        self.assert_request_id(rc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        self.assertRaises(exceptions.UnsupportedConsoleType, s.get_rdp_console, 'invalid')

    def test_get_console_url(self):
        s = self.cs.servers.get(1234)
        vc = s.get_console_url('novnc')
        self.assert_request_id(vc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        vc = self.cs.servers.get_console_url(s, 'novnc')
        self.assert_request_id(vc, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/remote-consoles')
        self.assertRaises(exceptions.UnsupportedConsoleType, s.get_console_url, 'invalid')
        self.assertRaises(exceptions.UnsupportedConsoleType, s.get_console_url, 'webmks')