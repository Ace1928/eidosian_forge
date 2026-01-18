import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_temp_failure(self):
    responses = [HTTPretty.Response(body="{'test': 'fail'}", status=500), HTTPretty.Response(body="{'test': 'success'}", status=200)]
    HTTPretty.register_uri(HTTPretty.POST, 'https://%s/temp_fail/' % self.region.endpoint, responses=responses)
    conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
    resp = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/temp_fail/', 'POST')
    self.assertEqual(resp.read(), b"{'test': 'success'}")