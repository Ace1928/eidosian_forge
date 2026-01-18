import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_port_pooling(self):
    conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret', port=8080)
    con1 = conn.get_http_connection(conn.host, conn.port, conn.is_secure)
    conn.put_http_connection(conn.host, conn.port, conn.is_secure, con1)
    con2 = conn.get_http_connection(conn.host, conn.port, conn.is_secure)
    conn.put_http_connection(conn.host, conn.port, conn.is_secure, con2)
    self.assertEqual(con1, con2)
    conn.port = 8081
    con3 = conn.get_http_connection(conn.host, conn.port, conn.is_secure)
    conn.put_http_connection(conn.host, conn.port, conn.is_secure, con3)
    self.assertNotEqual(con1, con3)