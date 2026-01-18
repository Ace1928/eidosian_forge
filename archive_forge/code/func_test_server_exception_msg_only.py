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
def test_server_exception_msg_only(self):
    error_msg = 'test error msg'
    error_body = _get_error_body(error_msg, err_type=ERROR_DICT)
    fake_resp = utils.FakeResponse({'content-type': 'application/json'}, io.StringIO(error_body), version=1, status=500)
    client = http.HTTPClient('http://localhost/')
    client.get_connection = lambda *a, **kw: utils.FakeConnection(fake_resp)
    error = self.assertRaises(exc.InternalServerError, client.json_request, 'GET', '/v1/resources')
    self.assertEqual(error_msg + ' (HTTP 500)', str(error))