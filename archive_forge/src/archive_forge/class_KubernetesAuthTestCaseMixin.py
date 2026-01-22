import os
import base64
from libcloud.utils.py3 import b
from libcloud.common.kubernetes import (
class KubernetesAuthTestCaseMixin:
    """
    Test class mixin which tests different type of Kubernetes authentication
    mechanisms (client cert, token, basic auth).

    It's to be used with all the drivers which inherit from KubernetesDriverMixin.
    """

    def test_http_basic_auth(self):
        driver = self.driver_cls(key='username', secret='password')
        self.assertEqual(driver.connectionCls, KubernetesBasicAuthConnection)
        self.assertEqual(driver.connection.user_id, 'username')
        self.assertEqual(driver.connection.key, 'password')
        auth_string = base64.b64encode(b('{}:{}'.format('username', 'password'))).decode('utf-8')
        headers = driver.connection.add_default_headers({})
        self.assertEqual(headers['Content-Type'], 'application/json')
        self.assertEqual(headers['Authorization'], 'Basic %s' % auth_string)

    def test_cert_auth(self):
        expected_msg = 'Both key and certificate files are needed'
        self.assertRaisesRegex(ValueError, expected_msg, self.driver_cls, key_file=KEY_FILE, ca_cert=CA_CERT_FILE)
        expected_msg = 'Both key and certificate files are needed'
        self.assertRaisesRegex(ValueError, expected_msg, self.driver_cls, cert_file=CERT_FILE, ca_cert=CA_CERT_FILE)
        driver = self.driver_cls(key_file=KEY_FILE, cert_file=CERT_FILE, ca_cert=CA_CERT_FILE)
        self.assertEqual(driver.connectionCls, KubernetesTLSAuthConnection)
        self.assertEqual(driver.connection.key_file, KEY_FILE)
        self.assertEqual(driver.connection.cert_file, CERT_FILE)
        self.assertEqual(driver.connection.connection.ca_cert, CA_CERT_FILE)
        headers = driver.connection.add_default_headers({})
        self.assertEqual(headers['Content-Type'], 'application/json')
        driver = self.driver_cls(key_file=KEY_FILE, cert_file=CERT_FILE, ca_cert=None)
        self.assertEqual(driver.connectionCls, KubernetesTLSAuthConnection)
        self.assertEqual(driver.connection.key_file, KEY_FILE)
        self.assertEqual(driver.connection.cert_file, CERT_FILE)
        self.assertEqual(driver.connection.connection.ca_cert, False)
        headers = driver.connection.add_default_headers({})
        self.assertEqual(headers['Content-Type'], 'application/json')

    def test_bearer_token_auth(self):
        driver = self.driver_cls(ex_token_bearer_auth=True, key='foobar')
        self.assertEqual(driver.connectionCls, KubernetesTokenAuthConnection)
        self.assertEqual(driver.connection.key, 'foobar')
        headers = driver.connection.add_default_headers({})
        self.assertEqual(headers['Content-Type'], 'application/json')
        self.assertEqual(headers['Authorization'], 'Bearer %s' % 'foobar')

    def test_host_sanitization(self):
        driver = self.driver_cls(host='example.com')
        self.assertEqual(driver.connection.host, 'example.com')
        driver = self.driver_cls(host='http://example.com')
        self.assertEqual(driver.connection.host, 'example.com')
        driver = self.driver_cls(host='https://example.com')
        self.assertEqual(driver.connection.host, 'example.com')