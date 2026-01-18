import socket
from unittest import mock
import io
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import testtools
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
from heatclient.tests.unit import fakes
def test_session_json_request(self):
    fake = fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'application/json'}, jsonutils.dumps({'some': 'body'}))
    self.request.return_value = (fake, {})
    client = http.SessionClient(session=mock.ANY, auth=mock.ANY)
    resp = client.request('', 'GET')
    self.assertEqual(200, resp.status_code)
    self.assertEqual({'some': 'body'}, resp.json())