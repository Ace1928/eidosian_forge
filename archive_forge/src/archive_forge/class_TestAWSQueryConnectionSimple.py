import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
class TestAWSQueryConnectionSimple(TestAWSQueryConnection):

    def test_query_connection_basis(self):
        HTTPretty.register_uri(HTTPretty.POST, 'https://%s/' % self.region.endpoint, json.dumps({'test': 'secure'}), content_type='application/json')
        conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
        self.assertEqual(conn.host, 'mockservice.cc-zone-1.amazonaws.com')

    def test_query_connection_noproxy(self):
        HTTPretty.register_uri(HTTPretty.POST, 'https://%s/' % self.region.endpoint, json.dumps({'test': 'secure'}), content_type='application/json')
        os.environ['no_proxy'] = self.region.endpoint
        conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret', proxy='NON_EXISTENT_HOSTNAME', proxy_port='3128')
        resp = conn.make_request('myCmd', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
        del os.environ['no_proxy']
        args = parse_qs(HTTPretty.last_request.body)
        self.assertEqual(args[b'AWSAccessKeyId'], [b'access_key'])

    def test_query_connection_noproxy_nosecure(self):
        HTTPretty.register_uri(HTTPretty.POST, 'https://%s/' % self.region.endpoint, json.dumps({'test': 'insecure'}), content_type='application/json')
        os.environ['no_proxy'] = self.region.endpoint
        conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret', proxy='NON_EXISTENT_HOSTNAME', proxy_port='3128', is_secure=False)
        resp = conn.make_request('myCmd', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
        del os.environ['no_proxy']
        args = parse_qs(HTTPretty.last_request.body)
        self.assertEqual(args[b'AWSAccessKeyId'], [b'access_key'])

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

    def test_multi_commands(self):
        """Check connection re-use"""
        HTTPretty.register_uri(HTTPretty.POST, 'https://%s/' % self.region.endpoint, json.dumps({'test': 'secure'}), content_type='application/json')
        conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
        resp1 = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
        body1 = parse_qs(HTTPretty.last_request.body)
        resp2 = conn.make_request('myCmd2', {'par3': 'bar', 'par4': 'narf'}, '/', 'POST')
        body2 = parse_qs(HTTPretty.last_request.body)
        self.assertEqual(body1[b'par1'], [b'foo'])
        self.assertEqual(body1[b'par2'], [b'baz'])
        with self.assertRaises(KeyError):
            body1[b'par3']
        self.assertEqual(body2[b'par3'], [b'bar'])
        self.assertEqual(body2[b'par4'], [b'narf'])
        with self.assertRaises(KeyError):
            body2['par1']
        self.assertEqual(resp1.read(), b'{"test": "secure"}')
        self.assertEqual(resp2.read(), b'{"test": "secure"}')

    def test_non_secure(self):
        HTTPretty.register_uri(HTTPretty.POST, 'http://%s/' % self.region.endpoint, json.dumps({'test': 'normal'}), content_type='application/json')
        conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret', is_secure=False)
        resp = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
        self.assertEqual(resp.read(), b'{"test": "normal"}')

    def test_alternate_port(self):
        HTTPretty.register_uri(HTTPretty.POST, 'http://%s:8080/' % self.region.endpoint, json.dumps({'test': 'alternate'}), content_type='application/json')
        conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret', port=8080, is_secure=False)
        resp = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
        self.assertEqual(resp.read(), b'{"test": "alternate"}')

    def test_temp_failure(self):
        responses = [HTTPretty.Response(body="{'test': 'fail'}", status=500), HTTPretty.Response(body="{'test': 'success'}", status=200)]
        HTTPretty.register_uri(HTTPretty.POST, 'https://%s/temp_fail/' % self.region.endpoint, responses=responses)
        conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
        resp = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/temp_fail/', 'POST')
        self.assertEqual(resp.read(), b"{'test': 'success'}")

    def test_unhandled_exception(self):
        HTTPretty.register_uri(HTTPretty.POST, 'https://%s/temp_exception/' % self.region.endpoint, responses=[])

        def fake_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
            raise socket.timeout('fake error')
        socket.create_connection = fake_connection
        conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
        conn.num_retries = 0
        with self.assertRaises(socket.error):
            resp = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/temp_exception/', 'POST')

    def test_connection_close(self):
        """Check connection re-use after close header is received"""
        HTTPretty.register_uri(HTTPretty.POST, 'https://%s/' % self.region.endpoint, json.dumps({'test': 'secure'}), content_type='application/json', connection='close')
        conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')

        def mock_put_conn(*args, **kwargs):
            raise Exception('put_http_connection should not be called!')
        conn.put_http_connection = mock_put_conn
        resp1 = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
        self.assertEqual(resp1.getheader('connection'), 'close')

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