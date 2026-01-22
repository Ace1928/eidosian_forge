from http import client as http_client
import io
from unittest import mock
from oslo_serialization import jsonutils
import socket
from magnumclient.common.apiclient.exceptions import GatewayTimeout
from magnumclient.common.apiclient.exceptions import MultipleChoices
from magnumclient.common import httpclient as http
from magnumclient import exceptions as exc
from magnumclient.tests import utils
class SessionClientTest(utils.BaseTestCase):

    def test_server_exception_msg_and_traceback(self):
        error_msg = 'another test error'
        error_trace = '"Traceback (most recent call last):\\n\\n  File \\"/usr/local/lib/python2.7/...'
        error_body = _get_error_body(error_msg, error_trace)
        fake_session = utils.FakeSession({'Content-Type': 'application/json'}, error_body, 500)
        client = http.SessionClient(session=fake_session)
        error = self.assertRaises(exc.InternalServerError, client.json_request, 'GET', '/v1/resources')
        self.assertEqual('%(error)s (HTTP 500)\n%(trace)s' % {'error': error_msg, 'trace': error_trace}, '%(error)s\n%(details)s' % {'error': str(error), 'details': str(error.details)})

    def test_server_exception_empty_body(self):
        error_body = _get_error_body()
        fake_session = utils.FakeSession({'Content-Type': 'application/json'}, error_body, 500)
        client = http.SessionClient(session=fake_session)
        error = self.assertRaises(exc.InternalServerError, client.json_request, 'GET', '/v1/resources')
        self.assertEqual('Internal Server Error (HTTP 500)', str(error))

    def test_bypass_url(self):
        fake_response = utils.FakeSessionResponse({}, content='', status_code=201)
        fake_session = mock.MagicMock()
        fake_session.request.side_effect = [fake_response]
        client = http.SessionClient(session=fake_session, endpoint_override='http://magnum')
        client.json_request('GET', '/v1/clusters')
        self.assertEqual(fake_session.request.call_args[1]['endpoint_override'], 'http://magnum')

    def test_exception(self):
        fake_response = utils.FakeSessionResponse({}, content='', status_code=504)
        fake_session = mock.MagicMock()
        fake_session.request.side_effect = [fake_response]
        client = http.SessionClient(session=fake_session, endpoint_override='http://magnum')
        self.assertRaises(GatewayTimeout, client.json_request, 'GET', '/v1/resources')

    def test_construct_http_client_return_httpclient(self):
        client = http._construct_http_client('http://localhost/')
        self.assertIsInstance(client, http.HTTPClient)

    def test_construct_http_client_return_sessionclient(self):
        fake_session = mock.MagicMock()
        client = http._construct_http_client(session=fake_session)
        self.assertIsInstance(client, http.SessionClient)

    def test_raw_request(self):
        fake_response = utils.FakeSessionResponse({'content-type': 'application/octet-stream'}, content='', status_code=200)
        fake_session = mock.MagicMock()
        fake_session.request.side_effect = [fake_response]
        client = http.SessionClient(session=fake_session, endpoint_override='http://magnum')
        resp, resp_body = client.raw_request('GET', '/v1/clusters')
        self.assertEqual(fake_session.request.call_args[1]['headers']['Content-Type'], 'application/octet-stream')
        self.assertEqual(None, resp_body)
        self.assertEqual(fake_response, resp)