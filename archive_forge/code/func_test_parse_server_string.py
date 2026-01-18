from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def test_parse_server_string(self):
    result = _utils.parse_server_string('::1')
    self.assertEqual(('::1', ''), result)
    result = _utils.parse_server_string('[::1]:8773')
    self.assertEqual(('::1', '8773'), result)
    result = _utils.parse_server_string('2001:db8::192.168.1.1')
    self.assertEqual(('2001:db8::192.168.1.1', ''), result)
    result = _utils.parse_server_string('[2001:db8::192.168.1.1]:8773')
    self.assertEqual(('2001:db8::192.168.1.1', '8773'), result)
    result = _utils.parse_server_string('192.168.1.1')
    self.assertEqual(('192.168.1.1', ''), result)
    result = _utils.parse_server_string('192.168.1.2:8773')
    self.assertEqual(('192.168.1.2', '8773'), result)
    result = _utils.parse_server_string('192.168.1.3')
    self.assertEqual(('192.168.1.3', ''), result)
    result = _utils.parse_server_string('www.example.com:8443')
    self.assertEqual(('www.example.com', '8443'), result)
    result = _utils.parse_server_string('www.example.com')
    self.assertEqual(('www.example.com', ''), result)
    result = _utils.parse_server_string('www.exa:mple.com:8443')
    self.assertEqual(('', ''), result)
    result = _utils.parse_server_string('')
    self.assertEqual(('', ''), result)