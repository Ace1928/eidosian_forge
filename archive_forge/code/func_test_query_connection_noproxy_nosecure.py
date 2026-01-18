import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_query_connection_noproxy_nosecure(self):
    HTTPretty.register_uri(HTTPretty.POST, 'https://%s/' % self.region.endpoint, json.dumps({'test': 'insecure'}), content_type='application/json')
    os.environ['no_proxy'] = self.region.endpoint
    conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret', proxy='NON_EXISTENT_HOSTNAME', proxy_port='3128', is_secure=False)
    resp = conn.make_request('myCmd', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
    del os.environ['no_proxy']
    args = parse_qs(HTTPretty.last_request.body)
    self.assertEqual(args[b'AWSAccessKeyId'], [b'access_key'])