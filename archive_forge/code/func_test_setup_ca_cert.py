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
def test_setup_ca_cert(self):
    self.httplib_object.verify = False
    self.httplib_object._setup_ca_cert()
    self.assertIsNone(self.httplib_object.ca_cert)
    self.httplib_object.verify = True
    libcloud.security.CA_CERTS_PATH = os.path.abspath(__file__)
    self.httplib_object._setup_ca_cert()
    self.assertTrue(self.httplib_object.ca_cert is not None)