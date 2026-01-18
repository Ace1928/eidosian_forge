import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_unhandled_exception(self):
    HTTPretty.register_uri(HTTPretty.POST, 'https://%s/temp_exception/' % self.region.endpoint, responses=[])

    def fake_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
        raise socket.timeout('fake error')
    socket.create_connection = fake_connection
    conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
    conn.num_retries = 0
    with self.assertRaises(socket.error):
        resp = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/temp_exception/', 'POST')