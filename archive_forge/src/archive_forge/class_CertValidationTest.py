import os
import ssl
import unittest
import mock
from nose.plugins.attrib import attr
import boto
from boto.pyami.config import Config
from boto import exception, https_connection
from boto.gs.connection import GSConnection
from boto.s3.connection import S3Connection
@attr('notdefault', 'ssl')
class CertValidationTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.config.add_section('Boto')
        self.config.setbool('Boto', 'https_validate_certificates', True)
        self.config.add_section('Credentials')
        self.config.set('Credentials', 'gs_access_key_id', 'xyz')
        self.config.set('Credentials', 'gs_secret_access_key', 'xyz')
        self.config.set('Credentials', 'aws_access_key_id', 'xyz')
        self.config.set('Credentials', 'aws_secret_access_key', 'xyz')
        self._config_patch = mock.patch('boto.config', self.config)
        self._config_patch.start()

    def tearDown(self):
        self._config_patch.stop()

    def enableProxy(self):
        self.config.set('Boto', 'proxy', PROXY_HOST)
        self.config.set('Boto', 'proxy_port', PROXY_PORT)

    def assertConnectionThrows(self, connection_class, error):
        conn = connection_class('fake_id', 'fake_secret')
        self.assertRaises(error, conn.get_all_buckets)

    def do_test_valid_cert(self):
        self.assertConnectionThrows(S3Connection, exception.S3ResponseError)
        self.assertConnectionThrows(GSConnection, exception.GSResponseError)

    def test_valid_cert(self):
        self.do_test_valid_cert()

    def test_valid_cert_with_proxy(self):
        self.enableProxy()
        self.do_test_valid_cert()

    def do_test_invalid_signature(self):
        self.config.set('Boto', 'ca_certificates_file', DEFAULT_CA_CERTS_FILE)
        self.assertConnectionThrows(S3Connection, ssl.SSLError)
        self.assertConnectionThrows(GSConnection, ssl.SSLError)

    def test_invalid_signature(self):
        self.do_test_invalid_signature()

    def test_invalid_signature_with_proxy(self):
        self.enableProxy()
        self.do_test_invalid_signature()

    def do_test_invalid_host(self):
        self.config.set('Credentials', 'gs_host', INVALID_HOSTNAME_HOST)
        self.config.set('Credentials', 's3_host', INVALID_HOSTNAME_HOST)
        self.assertConnectionThrows(S3Connection, ssl.SSLError)
        self.assertConnectionThrows(GSConnection, ssl.SSLError)

    def do_test_invalid_host(self):
        self.config.set('Credentials', 'gs_host', INVALID_HOSTNAME_HOST)
        self.config.set('Credentials', 's3_host', INVALID_HOSTNAME_HOST)
        self.assertConnectionThrows(S3Connection, https_connection.InvalidCertificateException)
        self.assertConnectionThrows(GSConnection, https_connection.InvalidCertificateException)

    def test_invalid_host(self):
        self.do_test_invalid_host()

    def test_invalid_host_with_proxy(self):
        self.enableProxy()
        self.do_test_invalid_host()