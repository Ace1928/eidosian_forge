import os
import ssl
import sys
import socket
from unittest import mock
from unittest.mock import Mock, patch
import requests_mock
from requests.exceptions import ConnectTimeout
import libcloud.common.base
from libcloud.http import LibcloudConnection, SignedHTTPSAdapter, LibcloudBaseConnection
from libcloud.test import unittest, no_internet
from libcloud.utils.py3 import assertRaisesRegex
from libcloud.common.base import Response, Connection, CertificateConnection
from libcloud.utils.retry import RETRY_EXCEPTIONS, Retry, RetryForeverOnRateLimitError
from libcloud.common.exceptions import RateLimitReachedError
def test_secure_connection_unusual_port(self):
    """
        Test that the connection class will default to secure (https) even
        when the port is an unusual (non 443, 80) number
        """
    conn = Connection(secure=True, host='localhost', port=8081)
    conn.connect()
    self.assertEqual(conn.connection.host, 'https://localhost:8081')
    conn2 = Connection(url='https://localhost:8081')
    conn2.connect()
    self.assertEqual(conn2.connection.host, 'https://localhost:8081')