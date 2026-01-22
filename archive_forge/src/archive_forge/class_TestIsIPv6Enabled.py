import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
class TestIsIPv6Enabled(test_base.BaseTestCase):

    def setUp(self):
        super(TestIsIPv6Enabled, self).setUp()

        def reset_detection_flag():
            netutils._IS_IPV6_ENABLED = None
        reset_detection_flag()
        self.addCleanup(reset_detection_flag)

    @mock.patch('os.path.exists', return_value=True)
    @mock.patch('builtins.open', return_value=mock_file_content('0'))
    def test_enabled(self, mock_open, exists):
        enabled = netutils.is_ipv6_enabled()
        self.assertTrue(enabled)

    @mock.patch('os.path.exists', return_value=True)
    @mock.patch('builtins.open', return_value=mock_file_content('1'))
    def test_disabled(self, mock_open, exists):
        enabled = netutils.is_ipv6_enabled()
        self.assertFalse(enabled)

    @mock.patch('os.path.exists', return_value=False)
    @mock.patch('builtins.open', side_effect=AssertionError('should not read'))
    def test_disabled_non_exists(self, mock_open, exists):
        enabled = netutils.is_ipv6_enabled()
        self.assertFalse(enabled)

    @mock.patch('os.path.exists', return_value=True)
    def test_memoize_enabled(self, exists):
        netutils._IS_IPV6_ENABLED = None
        with mock.patch('builtins.open', return_value=mock_file_content('0')) as mock_open:
            enabled = netutils.is_ipv6_enabled()
            self.assertTrue(mock_open.called)
            self.assertTrue(netutils._IS_IPV6_ENABLED)
            self.assertTrue(enabled)
        with mock.patch('builtins.open', side_effect=AssertionError('should not be called')):
            enabled = netutils.is_ipv6_enabled()
            self.assertTrue(enabled)

    @mock.patch('os.path.exists', return_value=True)
    def test_memoize_disabled(self, exists):
        netutils._IS_IPV6_ENABLED = None
        with mock.patch('builtins.open', return_value=mock_file_content('1')):
            enabled = netutils.is_ipv6_enabled()
            self.assertFalse(enabled)
        with mock.patch('builtins.open', side_effect=AssertionError('should not be called')):
            enabled = netutils.is_ipv6_enabled()
            self.assertFalse(enabled)

    @mock.patch('os.path.exists', return_value=False)
    @mock.patch('builtins.open', side_effect=AssertionError('should not read'))
    def test_memoize_not_exists(self, mock_open, exists):
        netutils._IS_IPV6_ENABLED = None
        enabled = netutils.is_ipv6_enabled()
        self.assertFalse(enabled)
        enabled = netutils.is_ipv6_enabled()
        self.assertFalse(enabled)