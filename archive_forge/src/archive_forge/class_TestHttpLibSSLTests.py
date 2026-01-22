import os
import sys
import time
import random
import os.path
import platform
import warnings
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import libcloud.security
from libcloud.http import LibcloudConnection
from libcloud.test import unittest, no_network
from libcloud.utils.py3 import reload, httplib, assertRaisesRegex
class TestHttpLibSSLTests(unittest.TestCase):

    def setUp(self):
        libcloud.security.VERIFY_SSL_CERT = False
        libcloud.security.CA_CERTS_PATH = ORIGINAL_CA_CERTS_PATH
        self.httplib_object = LibcloudConnection('foo.bar', port=80)

    def test_custom_ca_path_using_env_var_doesnt_exist(self):
        os.environ['SSL_CERT_FILE'] = '/foo/doesnt/exist'
        try:
            reload(libcloud.security)
        except ValueError as e:
            msg = "Certificate file /foo/doesnt/exist doesn't exist"
            self.assertEqual(str(e), msg)
        else:
            self.fail('Exception was not thrown')

    def test_custom_ca_path_using_env_var_is_directory(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.environ['SSL_CERT_FILE'] = file_path
        expected_msg = "Certificate file can't be a directory"
        assertRaisesRegex(self, ValueError, expected_msg, reload, libcloud.security)

    def test_custom_ca_path_using_env_var_exist(self):
        file_path = os.path.abspath(__file__)
        os.environ['SSL_CERT_FILE'] = file_path
        reload(libcloud.security)
        self.assertEqual(libcloud.security.CA_CERTS_PATH, file_path)

    def test_ca_cert_list_warning(self):
        with warnings.catch_warnings(record=True) as w:
            self.httplib_object.verify = True
            self.httplib_object._setup_ca_cert(ca_cert=[ORIGINAL_CA_CERTS_PATH])
            self.assertEqual(self.httplib_object.ca_cert, ORIGINAL_CA_CERTS_PATH)
            self.assertEqual(w[0].category, DeprecationWarning)

    def test_setup_ca_cert(self):
        self.httplib_object.verify = False
        self.httplib_object._setup_ca_cert()
        self.assertIsNone(self.httplib_object.ca_cert)
        self.httplib_object.verify = True
        libcloud.security.CA_CERTS_PATH = os.path.abspath(__file__)
        self.httplib_object._setup_ca_cert()
        self.assertTrue(self.httplib_object.ca_cert is not None)