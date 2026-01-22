from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
class VersionNegotiationMixinTest(utils.BaseTestCase):

    def setUp(self):
        super(VersionNegotiationMixinTest, self).setUp()
        self.test_object = http.VersionNegotiationMixin()
        self.test_object.os_ironic_api_version = '1.6'
        self.test_object.api_version_select_state = 'default'
        self.test_object.endpoint_override = 'http://localhost:1234'
        self.mock_mcu = mock.MagicMock()
        self.test_object._make_connection_url = self.mock_mcu
        self.response = utils.FakeResponse({}, status=http_client.NOT_ACCEPTABLE)
        self.test_object.get_server = mock.MagicMock(return_value=('localhost', '1234'))

    def test__generic_parse_version_headers_has_headers(self):
        response = {'X-OpenStack-Ironic-API-Minimum-Version': '1.1', 'X-OpenStack-Ironic-API-Maximum-Version': '1.6'}
        expected = ('1.1', '1.6')
        result = self.test_object._generic_parse_version_headers(response.get)
        self.assertEqual(expected, result)

    def test__generic_parse_version_headers_missing_headers(self):
        response = {}
        expected = (None, None)
        result = self.test_object._generic_parse_version_headers(response.get)
        self.assertEqual(expected, result)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    def test_negotiate_version_bad_state(self, mock_save_data):
        self.test_object.api_version_select_state = 'word of the day: augur'
        self.assertRaises(RuntimeError, self.test_object.negotiate_version, None, None)
        self.assertEqual(0, mock_save_data.call_count)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_older(self, mock_pvh, mock_save_data):
        latest_ver = '1.5'
        mock_pvh.return_value = ('1.1', latest_ver)
        mock_conn = mock.MagicMock()
        result = self.test_object.negotiate_version(mock_conn, self.response)
        self.assertEqual(latest_ver, result)
        self.assertEqual(1, mock_pvh.call_count)
        host, port = http.get_server(self.test_object.endpoint_override)
        mock_save_data.assert_called_once_with(host=host, port=port, data=latest_ver)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_newer(self, mock_pvh, mock_save_data):
        mock_pvh.return_value = ('1.1', '1.10')
        mock_conn = mock.MagicMock()
        result = self.test_object.negotiate_version(mock_conn, self.response)
        self.assertEqual('1.6', result)
        self.assertEqual(1, mock_pvh.call_count)
        mock_save_data.assert_called_once_with(host=DEFAULT_HOST, port=DEFAULT_PORT, data='1.6')

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_make_simple_request', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_no_version_on_error(self, mock_pvh, mock_msr, mock_save_data):
        mock_pvh.side_effect = iter([(None, None), ('1.1', '1.2')])
        mock_conn = mock.MagicMock()
        result = self.test_object.negotiate_version(mock_conn, self.response)
        self.assertEqual('1.2', result)
        self.assertTrue(mock_msr.called)
        self.assertEqual(2, mock_pvh.call_count)
        self.assertEqual(1, mock_save_data.call_count)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_explicit_too_high(self, mock_pvh, mock_save_data):
        mock_pvh.return_value = ('1.1', '1.6')
        mock_conn = mock.MagicMock()
        self.test_object.api_version_select_state = 'user'
        self.test_object.os_ironic_api_version = '99.99'
        self.assertRaises(exc.UnsupportedVersion, self.test_object.negotiate_version, mock_conn, self.response)
        self.assertEqual(1, mock_pvh.call_count)
        self.assertEqual(0, mock_save_data.call_count)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_explicit_not_supported(self, mock_pvh, mock_save_data):
        mock_pvh.return_value = ('1.1', '1.6')
        mock_conn = mock.MagicMock()
        self.test_object.api_version_select_state = 'negotiated'
        self.test_object.os_ironic_api_version = '1.5'
        self.assertRaises(exc.UnsupportedVersion, self.test_object.negotiate_version, mock_conn, self.response)
        self.assertEqual(1, mock_pvh.call_count)
        self.assertEqual(0, mock_save_data.call_count)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_strict_version_comparison(self, mock_pvh, mock_save_data):
        max_ver = '1.10'
        mock_pvh.return_value = ('1.2', max_ver)
        mock_conn = mock.MagicMock()
        self.test_object.os_ironic_api_version = '1.10'
        result = self.test_object.negotiate_version(mock_conn, self.response)
        self.assertEqual(max_ver, result)
        self.assertEqual(1, mock_pvh.call_count)
        host, port = http.get_server(self.test_object.endpoint_override)
        mock_save_data.assert_called_once_with(host=host, port=port, data=max_ver)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_make_simple_request', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_user_latest(self, mock_pvh, mock_msr, mock_save_data):
        mock_pvh.side_effect = iter([(None, None), ('1.1', '1.99')])
        mock_conn = mock.MagicMock()
        self.test_object.api_version_select_state = 'user'
        self.test_object.os_ironic_api_version = 'latest'
        result = self.test_object.negotiate_version(mock_conn, None)
        self.assertEqual(http.LATEST_VERSION, result)
        self.assertEqual('negotiated', self.test_object.api_version_select_state)
        self.assertEqual(http.LATEST_VERSION, self.test_object.os_ironic_api_version)
        self.assertTrue(mock_msr.called)
        self.assertEqual(2, mock_pvh.call_count)
        self.assertEqual(1, mock_save_data.call_count)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_make_simple_request', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_user_list(self, mock_pvh, mock_msr, mock_save_data):
        mock_pvh.side_effect = [(None, None), ('1.1', '1.26')]
        mock_conn = mock.MagicMock()
        self.test_object.api_version_select_state = 'user'
        self.test_object.os_ironic_api_version = ['1.1', '1.6', '1.25', '1.26', '1.26.1', '1.27', '1.30']
        result = self.test_object.negotiate_version(mock_conn, self.response)
        self.assertEqual('1.26', result)
        self.assertEqual('negotiated', self.test_object.api_version_select_state)
        self.assertEqual('1.26', self.test_object.os_ironic_api_version)
        self.assertTrue(mock_msr.called)
        self.assertEqual(2, mock_pvh.call_count)
        self.assertEqual(1, mock_save_data.call_count)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_make_simple_request', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_user_list_fails_nomatch(self, mock_pvh, mock_msr, mock_save_data):
        mock_pvh.side_effect = iter([(None, None), ('1.2', '1.26')])
        mock_conn = mock.MagicMock()
        self.test_object.api_version_select_state = 'user'
        self.test_object.os_ironic_api_version = ['1.39', '1.1']
        self.assertRaises(exc.UnsupportedVersion, self.test_object.negotiate_version, mock_conn, self.response)
        self.assertEqual('user', self.test_object.api_version_select_state)
        self.assertEqual(['1.39', '1.1'], self.test_object.os_ironic_api_version)
        self.assertEqual(2, mock_pvh.call_count)
        self.assertEqual(0, mock_save_data.call_count)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_make_simple_request', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_user_list_single_value(self, mock_pvh, mock_msr, mock_save_data):
        mock_pvh.side_effect = iter([(None, None), ('1.1', '1.26')])
        mock_conn = mock.MagicMock()
        self.test_object.api_version_select_state = 'user'
        self.test_object.os_ironic_api_version = ['1.01']
        result = self.test_object.negotiate_version(mock_conn, None)
        self.assertEqual('1.1', result)
        self.assertEqual('negotiated', self.test_object.api_version_select_state)
        self.assertEqual('1.1', self.test_object.os_ironic_api_version)
        self.assertTrue(mock_msr.called)
        self.assertEqual(2, mock_pvh.call_count)
        self.assertEqual(1, mock_save_data.call_count)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_make_simple_request', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_server_user_list_fails_latest(self, mock_pvh, mock_msr, mock_save_data):
        mock_pvh.side_effect = iter([(None, None), ('1.1', '1.2')])
        mock_conn = mock.MagicMock()
        self.test_object.api_version_select_state = 'user'
        self.test_object.os_ironic_api_version = ['1.01', 'latest']
        self.assertRaises(ValueError, self.test_object.negotiate_version, mock_conn, self.response)
        self.assertEqual('user', self.test_object.api_version_select_state)
        self.assertEqual(['1.01', 'latest'], self.test_object.os_ironic_api_version)
        self.assertEqual(2, mock_pvh.call_count)
        self.assertEqual(0, mock_save_data.call_count)

    @mock.patch.object(filecache, 'save_data', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_make_simple_request', autospec=True)
    @mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
    def test_negotiate_version_explicit_version_request(self, mock_pvh, mock_msr, mock_save_data):
        mock_pvh.side_effect = iter([(None, None), ('1.1', '1.99')])
        mock_conn = mock.MagicMock()
        self.test_object.api_version_select_state = 'negotiated'
        self.test_object.os_ironic_api_version = '1.30'
        req_header = {'X-OpenStack-Ironic-API-Version': '1.29'}
        response = utils.FakeResponse({}, status=http_client.NOT_ACCEPTABLE, request_headers=req_header)
        self.assertRaisesRegex(exc.UnsupportedVersion, '.*is not supported by the server.*', self.test_object.negotiate_version, mock_conn, response)
        self.assertTrue(mock_msr.called)
        self.assertEqual(2, mock_pvh.call_count)
        self.assertFalse(mock_save_data.called)

    def test_get_server(self):
        host = 'ironic-host'
        port = '6385'
        endpoint_override = 'http://%s:%s/ironic/v1/' % (host, port)
        self.assertEqual((host, port), http.get_server(endpoint_override))