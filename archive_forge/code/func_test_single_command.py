import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_single_command(self):
    HTTPretty.register_uri(HTTPretty.POST, 'https://%s/' % self.region.endpoint, json.dumps({'test': 'secure'}), content_type='application/json')
    conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
    resp = conn.make_request('myCmd', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
    args = parse_qs(HTTPretty.last_request.body)
    self.assertEqual(args[b'AWSAccessKeyId'], [b'access_key'])
    self.assertEqual(args[b'SignatureMethod'], [b'HmacSHA256'])
    self.assertEqual(args[b'Version'], [conn.APIVersion.encode('utf-8')])
    self.assertEqual(args[b'par1'], [b'foo'])
    self.assertEqual(args[b'par2'], [b'baz'])
    self.assertEqual(resp.read(), b'{"test": "secure"}')