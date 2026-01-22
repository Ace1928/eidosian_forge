import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class FileHandleTest(base.TestCase):
    """Tests for FileHandle."""

    def test_close(self):
        file_handle = mock.Mock()
        vmw_http_file = rw_handles.FileHandle(file_handle)
        vmw_http_file.close()
        file_handle.close.assert_called_once_with()

    @mock.patch('urllib3.connection.HTTPConnection')
    def test_create_connection_http(self, http_conn):
        conn = mock.Mock()
        http_conn.return_value = conn
        handle = rw_handles.FileHandle(None)
        ret = handle._create_connection('http://localhost/foo?q=bar', 'GET')
        self.assertEqual(conn, ret)
        conn.putrequest.assert_called_once_with('GET', '/foo?q=bar')

    @mock.patch('urllib3.connection.HTTPSConnection')
    def test_create_connection_https(self, https_conn):
        conn = mock.Mock()
        https_conn.return_value = conn
        handle = rw_handles.FileHandle(None)
        ret = handle._create_connection('https://localhost/foo?q=bar', 'GET')
        self.assertEqual(conn, ret)
        ca_store = requests.certs.where()
        conn.set_cert.assert_called_once_with(ca_certs=ca_store, cert_reqs=ssl.CERT_NONE, assert_fingerprint=None)
        conn.putrequest.assert_called_once_with('GET', '/foo?q=bar')

    @mock.patch('urllib3.connection.HTTPSConnection')
    def test_create_connection_https_with_cacerts(self, https_conn):
        conn = mock.Mock()
        https_conn.return_value = conn
        handle = rw_handles.FileHandle(None)
        ret = handle._create_connection('https://localhost/foo?q=bar', 'GET', cacerts=True)
        self.assertEqual(conn, ret)
        ca_store = requests.certs.where()
        conn.set_cert.assert_called_once_with(ca_certs=ca_store, cert_reqs=ssl.CERT_REQUIRED, assert_fingerprint=None)

    @mock.patch('urllib3.connection.HTTPSConnection')
    def test_create_connection_https_with_ssl_thumbprint(self, https_conn):
        conn = mock.Mock()
        https_conn.return_value = conn
        handle = rw_handles.FileHandle(None)
        cacerts = mock.sentinel.cacerts
        thumbprint = mock.sentinel.thumbprint
        ret = handle._create_connection('https://localhost/foo?q=bar', 'GET', cacerts=cacerts, ssl_thumbprint=thumbprint)
        self.assertEqual(conn, ret)
        conn.set_cert.assert_called_once_with(ca_certs=cacerts, cert_reqs=None, assert_fingerprint=thumbprint)