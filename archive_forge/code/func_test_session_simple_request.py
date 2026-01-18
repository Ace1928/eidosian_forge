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
def test_session_simple_request(self):
    resp = fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'application/octet-stream'}, '')
    self.request.return_value = (resp, '')
    client = http.SessionClient(session=mock.ANY, auth=mock.ANY)
    response = client.request(method='GET', url='')
    self.assertEqual(200, response.status_code)
    self.assertEqual('', ''.join([x for x in response.content]))