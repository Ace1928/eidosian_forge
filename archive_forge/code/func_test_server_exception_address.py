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
def test_server_exception_address(self):
    endpoint = 'https://magnum-host:6385'
    client = http.HTTPClient(endpoint, token='foobar', insecure=True, ca_file='/path/to/ca_file')
    client.get_connection = lambda *a, **kw: utils.FakeConnection(exc=socket.gaierror)
    self.assertRaises(exc.EndpointNotFound, client.json_request, 'GET', '/v1/resources', body='farboo')