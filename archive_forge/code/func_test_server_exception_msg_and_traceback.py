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
def test_server_exception_msg_and_traceback(self):
    error_msg = 'another test error'
    error_trace = '"Traceback (most recent call last):\\n\\n  File \\"/usr/local/lib/python2.7/...'
    error_body = _get_error_body(error_msg, error_trace)
    fake_session = utils.FakeSession({'Content-Type': 'application/json'}, error_body, 500)
    client = http.SessionClient(session=fake_session)
    error = self.assertRaises(exc.InternalServerError, client.json_request, 'GET', '/v1/resources')
    self.assertEqual('%(error)s (HTTP 500)\n%(trace)s' % {'error': error_msg, 'trace': error_trace}, '%(error)s\n%(details)s' % {'error': str(error), 'details': str(error.details)})