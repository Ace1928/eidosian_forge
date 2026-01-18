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
def test_passed_cert_to_verify_cert(self, mock_request):
    client = http.HTTPClient('https://foo', ca_file='NOWHERE')
    self.assertEqual('NOWHERE', client.verify_cert)
    with mock.patch('heatclient.common.http.get_system_ca_file') as gsf:
        gsf.return_value = 'SOMEWHERE'
        client = http.HTTPClient('https://foo')
        self.assertEqual('SOMEWHERE', client.verify_cert)