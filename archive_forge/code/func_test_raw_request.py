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
def test_raw_request(self):
    fake_response = utils.FakeSessionResponse({'content-type': 'application/octet-stream'}, content='', status_code=200)
    fake_session = mock.MagicMock()
    fake_session.request.side_effect = [fake_response]
    client = http.SessionClient(session=fake_session, endpoint_override='http://magnum')
    resp, resp_body = client.raw_request('GET', '/v1/clusters')
    self.assertEqual(fake_session.request.call_args[1]['headers']['Content-Type'], 'application/octet-stream')
    self.assertEqual(None, resp_body)
    self.assertEqual(fake_response, resp)