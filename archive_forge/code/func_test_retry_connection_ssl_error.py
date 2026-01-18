import ssl
import socket
from unittest.mock import Mock, MagicMock, patch
from libcloud.test import unittest
from libcloud.common.base import Connection
from libcloud.utils.retry import TRANSIENT_SSL_ERROR
def test_retry_connection_ssl_error(self):
    conn = Connection(timeout=0.2, retry_delay=0.1)
    with patch.object(conn, 'connect', Mock()):
        with patch.object(conn, 'connection') as connection:
            connection.request = MagicMock(__name__='request', side_effect=ssl.SSLError(TRANSIENT_SSL_ERROR))
            self.assertRaises(ssl.SSLError, conn.request, '/')
            self.assertGreater(connection.request.call_count, 1)