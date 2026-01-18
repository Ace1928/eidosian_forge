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
@mock.patch('os.path.getsize')
def test_send_with_local_file_url(self, get_size_mock):
    transport = service.RequestsTransport()
    url = 'file:///foo'
    request = requests.Request('GET', url).prepare()
    data = b'Hello World'
    get_size_mock.return_value = len(data)

    def read_mock():
        return data
    open_mock = mock.MagicMock(name='file_handle', spec=open)
    file_spec = list(set(dir(io.TextIOWrapper)).union(set(dir(io.BytesIO))))
    file_handle = mock.MagicMock(spec=file_spec)
    file_handle.write.return_value = None
    file_handle.__enter__.return_value = file_handle
    file_handle.read.side_effect = read_mock
    open_mock.return_value = file_handle
    with mock.patch('builtins.open', open_mock, create=True):
        resp = transport.session.send(request)
        self.assertEqual(data, resp.content)