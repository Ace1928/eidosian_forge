import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_write_post(self):
    session = self._create_mock_session()
    handle = rw_handles.VmdkWriteHandle(session, '10.1.2.3', 443, 'rp-1', 'folder-1', None, 100, http_method='POST')
    data = [1] * 10
    handle.write(data)
    self.assertEqual(len(data), handle._bytes_written)
    self._conn.putrequest.assert_called_once_with('POST', '/ds/disk1.vmdk')
    self._conn.send.assert_called_once_with(data)