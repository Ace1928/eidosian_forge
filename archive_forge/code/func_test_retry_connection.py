import ssl
import socket
from unittest.mock import Mock, MagicMock, patch
from libcloud.test import unittest
from libcloud.common.base import Connection
from libcloud.utils.retry import TRANSIENT_SSL_ERROR
def test_retry_connection(self):
    con = Connection(timeout=0.2, retry_delay=0.1)
    con.connection = Mock()
    connect_method = 'libcloud.common.base.Connection.request'
    with patch(connect_method) as mock_connect:
        try:
            mock_connect.side_effect = socket.gaierror('')
            con.request('/')
        except socket.gaierror:
            pass