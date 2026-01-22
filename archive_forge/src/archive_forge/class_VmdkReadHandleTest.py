import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class VmdkReadHandleTest(base.TestCase):
    """Tests for VmdkReadHandle."""

    def setUp(self):
        super(VmdkReadHandleTest, self).setUp()

    def _mock_connection(self, read_data='fake-data'):
        self._resp = mock.Mock()
        self._resp.read.return_value = read_data
        self._conn = mock.Mock()
        self._conn.getresponse.return_value = self._resp
        patcher = mock.patch('urllib3.connection.HTTPConnection')
        self.addCleanup(patcher.stop)
        HTTPConnectionMock = patcher.start()
        HTTPConnectionMock.return_value = self._conn

    def _create_mock_session(self, disk=True, progress=-1, read_data='fake-data'):
        self._mock_connection(read_data=read_data)
        device_url = mock.Mock()
        device_url.disk = disk
        device_url.url = 'http://*/ds/disk1.vmdk'
        lease_info = mock.Mock()
        lease_info.deviceUrl = [device_url]
        session = mock.Mock()

        def session_invoke_api_side_effect(module, method, *args, **kwargs):
            if module == session.vim:
                if method == 'ExportVm':
                    return mock.Mock()
                elif method == 'HttpNfcLeaseProgress':
                    self.assertEqual(progress, kwargs['percent'])
                    return
            return lease_info
        session.invoke_api.side_effect = session_invoke_api_side_effect
        vim_cookie = mock.Mock()
        vim_cookie.name = 'name'
        vim_cookie.value = 'value'
        session.vim.client.cookiejar = [vim_cookie]
        return session

    def test_init_failure(self):
        session = self._create_mock_session(False)
        self.assertRaises(exceptions.VimException, rw_handles.VmdkReadHandle, session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', 100)

    def test_read(self):
        chunk_size = rw_handles.READ_CHUNKSIZE
        session = self._create_mock_session()
        handle = rw_handles.VmdkReadHandle(session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', chunk_size * 10)
        fake_data = 'fake-data'
        data = handle.read(chunk_size)
        self.assertEqual(fake_data, data)
        self.assertEqual(len(fake_data), handle._bytes_read)

    def test_read_small(self):
        read_data = 'fake'
        session = self._create_mock_session(read_data=read_data)
        read_size = len(read_data)
        handle = rw_handles.VmdkReadHandle(session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', read_size * 10)
        handle.read(read_size)
        self.assertEqual(read_size, handle._bytes_read)

    def test_tell(self):
        chunk_size = rw_handles.READ_CHUNKSIZE
        session = self._create_mock_session()
        handle = rw_handles.VmdkReadHandle(session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', chunk_size * 10)
        data = handle.read(chunk_size)
        self.assertEqual(len(data), handle.tell())

    def test_update_progress(self):
        chunk_size = len('fake-data')
        vmdk_size = chunk_size * 10
        session = self._create_mock_session(True, 10)
        handle = rw_handles.VmdkReadHandle(session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', vmdk_size)
        data = handle.read(chunk_size)
        handle.update_progress()
        self.assertEqual('fake-data', data)

    def test_close(self):
        session = self._create_mock_session()
        handle = rw_handles.VmdkReadHandle(session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', 100)

        def session_invoke_api_side_effect(module, method, *args, **kwargs):
            if module == vim_util and method == 'get_object_property':
                return 'ready'
            self.assertEqual(session.vim, module)
            self.assertEqual('HttpNfcLeaseComplete', method)
        session.invoke_api = mock.Mock(side_effect=session_invoke_api_side_effect)
        handle._get_progress = mock.Mock(return_value=100)
        handle.close()
        self.assertEqual(2, session.invoke_api.call_count)

    def test_close_with_error(self):
        session = self._create_mock_session()
        handle = rw_handles.VmdkReadHandle(session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', 100)
        session.invoke_api.side_effect = exceptions.VimException(None)
        self.assertRaises(exceptions.VimException, handle.close)
        self._resp.close.assert_called_once_with()