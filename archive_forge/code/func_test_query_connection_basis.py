import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_query_connection_basis(self):
    HTTPretty.register_uri(HTTPretty.POST, 'https://%s/' % self.region.endpoint, json.dumps({'test': 'secure'}), content_type='application/json')
    conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
    self.assertEqual(conn.host, 'mockservice.cc-zone-1.amazonaws.com')