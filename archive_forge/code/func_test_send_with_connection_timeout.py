import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_send_with_connection_timeout(self):
    transport = service.RequestsTransport(connection_timeout=120)
    request = mock.Mock(url=mock.sentinel.url, message=mock.sentinel.message, headers=mock.sentinel.req_headers)
    with mock.patch.object(transport.session, 'post') as mock_post:
        transport.send(request)
        mock_post.assert_called_once_with(mock.sentinel.url, data=mock.sentinel.message, headers=mock.sentinel.req_headers, timeout=120, verify=transport.verify)