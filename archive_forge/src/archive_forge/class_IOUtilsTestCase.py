from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@ddt.ddt
class IOUtilsTestCase(test_base.BaseTestCase):
    _autospec_classes = [ioutils.win32utils.Win32Utils]

    def setUp(self):
        super(IOUtilsTestCase, self).setUp()
        self._setup_lib_mocks()
        self._ioutils = ioutils.IOUtils()
        self._mock_run = self._ioutils._win32_utils.run_and_check_output
        self._run_args = dict(kernel32_lib_func=True, failure_exc=exceptions.Win32IOException, eventlet_nonblocking_mode=False)
        self.addCleanup(mock.patch.stopall)

    def _setup_lib_mocks(self):
        self._ctypes = mock.Mock()
        self._ctypes.byref = lambda x: (x, 'byref')
        self._ctypes.c_wchar_p = lambda x: (x, 'c_wchar_p')
        mock.patch.multiple(ioutils, ctypes=self._ctypes, kernel32=mock.DEFAULT, create=True).start()

    def test_run_and_check_output(self):
        ret_val = self._ioutils._run_and_check_output(mock.sentinel.func, mock.sentinel.arg)
        self._mock_run.assert_called_once_with(mock.sentinel.func, mock.sentinel.arg, **self._run_args)
        self.assertEqual(self._mock_run.return_value, ret_val)

    @ddt.data({}, {'inherit_handle': True}, {'sec_attr': mock.sentinel.sec_attr})
    @ddt.unpack
    @mock.patch.object(wintypes, 'HANDLE')
    @mock.patch.object(wintypes, 'SECURITY_ATTRIBUTES')
    def test_create_pipe(self, mock_sec_attr_cls, mock_handle_cls, inherit_handle=False, sec_attr=None):
        r, w = self._ioutils.create_pipe(sec_attr, mock.sentinel.size, inherit_handle)
        exp_sec_attr = None
        if sec_attr:
            exp_sec_attr = sec_attr
        elif inherit_handle:
            exp_sec_attr = mock_sec_attr_cls.return_value
        self.assertEqual(mock_handle_cls.return_value.value, r)
        self.assertEqual(mock_handle_cls.return_value.value, w)
        self._mock_run.assert_called_once_with(ioutils.kernel32.CreatePipe, self._ctypes.byref(mock_handle_cls.return_value), self._ctypes.byref(mock_handle_cls.return_value), self._ctypes.byref(exp_sec_attr) if exp_sec_attr else None, mock.sentinel.size, **self._run_args)
        if not sec_attr and exp_sec_attr:
            self.assertEqual(inherit_handle, exp_sec_attr.bInheritHandle)
            self.assertEqual(self._ctypes.sizeof.return_value, exp_sec_attr.nLength)
            self._ctypes.sizeof.assert_called_once_with(exp_sec_attr)

    def test_wait_named_pipe(self):
        fake_timeout_s = 10
        self._ioutils.wait_named_pipe(mock.sentinel.pipe_name, timeout=fake_timeout_s)
        self._mock_run.assert_called_once_with(ioutils.kernel32.WaitNamedPipeW, self._ctypes.c_wchar_p(mock.sentinel.pipe_name), fake_timeout_s * 1000, **self._run_args)

    def test_open(self):
        handle = self._ioutils.open(mock.sentinel.path, mock.sentinel.access, mock.sentinel.share_mode, mock.sentinel.create_disposition, mock.sentinel.flags)
        self._mock_run.assert_called_once_with(ioutils.kernel32.CreateFileW, self._ctypes.c_wchar_p(mock.sentinel.path), mock.sentinel.access, mock.sentinel.share_mode, None, mock.sentinel.create_disposition, mock.sentinel.flags, None, error_ret_vals=[w_const.INVALID_HANDLE_VALUE], **self._run_args)
        self.assertEqual(self._mock_run.return_value, handle)

    def test_cancel_io(self):
        self._ioutils.cancel_io(mock.sentinel.handle, mock.sentinel.overlapped_struct, ignore_invalid_handle=True)
        expected_ignored_err_codes = [w_const.ERROR_NOT_FOUND, w_const.ERROR_INVALID_HANDLE]
        self._mock_run.assert_called_once_with(ioutils.kernel32.CancelIoEx, mock.sentinel.handle, self._ctypes.byref(mock.sentinel.overlapped_struct), ignored_error_codes=expected_ignored_err_codes, **self._run_args)

    def test_close_handle(self):
        self._ioutils.close_handle(mock.sentinel.handle)
        self._mock_run.assert_called_once_with(ioutils.kernel32.CloseHandle, mock.sentinel.handle, **self._run_args)

    def test_wait_io_completion(self):
        self._ioutils._wait_io_completion(mock.sentinel.event)
        self._mock_run.assert_called_once_with(ioutils.kernel32.WaitForSingleObjectEx, mock.sentinel.event, ioutils.WAIT_INFINITE_TIMEOUT, True, error_ret_vals=[w_const.WAIT_FAILED], **self._run_args)

    def test_set_event(self):
        self._ioutils.set_event(mock.sentinel.event)
        self._mock_run.assert_called_once_with(ioutils.kernel32.SetEvent, mock.sentinel.event, **self._run_args)

    def test_reset_event(self):
        self._ioutils._reset_event(mock.sentinel.event)
        self._mock_run.assert_called_once_with(ioutils.kernel32.ResetEvent, mock.sentinel.event, **self._run_args)

    def test_create_event(self):
        event = self._ioutils._create_event(mock.sentinel.event_attributes, mock.sentinel.manual_reset, mock.sentinel.initial_state, mock.sentinel.name)
        self._mock_run.assert_called_once_with(ioutils.kernel32.CreateEventW, mock.sentinel.event_attributes, mock.sentinel.manual_reset, mock.sentinel.initial_state, mock.sentinel.name, error_ret_vals=[None], **self._run_args)
        self.assertEqual(self._mock_run.return_value, event)

    @mock.patch.object(wintypes, 'LPOVERLAPPED', create=True)
    @mock.patch.object(wintypes, 'LPOVERLAPPED_COMPLETION_ROUTINE', lambda x: x, create=True)
    @mock.patch.object(ioutils.IOUtils, 'set_event')
    def test_get_completion_routine(self, mock_set_event, mock_LPOVERLAPPED):
        mock_callback = mock.Mock()
        compl_routine = self._ioutils.get_completion_routine(mock_callback)
        compl_routine(mock.sentinel.error_code, mock.sentinel.num_bytes, mock.sentinel.lpOverLapped)
        self._ctypes.cast.assert_called_once_with(mock.sentinel.lpOverLapped, wintypes.LPOVERLAPPED)
        mock_overlapped_struct = self._ctypes.cast.return_value.contents
        mock_set_event.assert_called_once_with(mock_overlapped_struct.hEvent)
        mock_callback.assert_called_once_with(mock.sentinel.num_bytes)

    @mock.patch.object(wintypes, 'OVERLAPPED', create=True)
    @mock.patch.object(ioutils.IOUtils, '_create_event')
    def test_get_new_overlapped_structure(self, mock_create_event, mock_OVERLAPPED):
        overlapped_struct = self._ioutils.get_new_overlapped_structure()
        self.assertEqual(mock_OVERLAPPED.return_value, overlapped_struct)
        self.assertEqual(mock_create_event.return_value, overlapped_struct.hEvent)

    @mock.patch.object(ioutils.IOUtils, '_reset_event')
    @mock.patch.object(ioutils.IOUtils, '_wait_io_completion')
    def test_read(self, mock_wait_io_completion, mock_reset_event):
        mock_overlapped_struct = mock.Mock()
        mock_event = mock_overlapped_struct.hEvent
        self._ioutils.read(mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, mock_overlapped_struct, mock.sentinel.compl_routine)
        mock_reset_event.assert_called_once_with(mock_event)
        self._mock_run.assert_called_once_with(ioutils.kernel32.ReadFileEx, mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, self._ctypes.byref(mock_overlapped_struct), mock.sentinel.compl_routine, **self._run_args)
        mock_wait_io_completion.assert_called_once_with(mock_event)

    @mock.patch.object(wintypes, 'DWORD')
    def test_read_file(self, mock_dword):
        num_bytes_read = mock_dword.return_value
        ret_val = self._ioutils.read_file(mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, mock.sentinel.overlapped_struct)
        self.assertEqual(num_bytes_read.value, ret_val)
        self._mock_run.assert_called_once_with(ioutils.kernel32.ReadFile, mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, self._ctypes.byref(num_bytes_read), self._ctypes.byref(mock.sentinel.overlapped_struct), **self._run_args)

    @mock.patch.object(ioutils.IOUtils, '_reset_event')
    @mock.patch.object(ioutils.IOUtils, '_wait_io_completion')
    def test_write(self, mock_wait_io_completion, mock_reset_event):
        mock_overlapped_struct = mock.Mock()
        mock_event = mock_overlapped_struct.hEvent
        self._ioutils.write(mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, mock_overlapped_struct, mock.sentinel.compl_routine)
        mock_reset_event.assert_called_once_with(mock_event)
        self._mock_run.assert_called_once_with(ioutils.kernel32.WriteFileEx, mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, self._ctypes.byref(mock_overlapped_struct), mock.sentinel.compl_routine, **self._run_args)
        mock_wait_io_completion.assert_called_once_with(mock_event)

    @mock.patch.object(wintypes, 'DWORD')
    def test_write_file(self, mock_dword):
        num_bytes_written = mock_dword.return_value
        ret_val = self._ioutils.write_file(mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, mock.sentinel.overlapped_struct)
        self.assertEqual(num_bytes_written.value, ret_val)
        self._mock_run.assert_called_once_with(ioutils.kernel32.WriteFile, mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, self._ctypes.byref(num_bytes_written), self._ctypes.byref(mock.sentinel.overlapped_struct), **self._run_args)

    def test_buffer_ops(self):
        mock.patch.stopall()
        fake_data = 'fake data'
        buff = self._ioutils.get_buffer(len(fake_data), data=fake_data)
        buff_data = self._ioutils.get_buffer_data(buff, len(fake_data))
        self.assertEqual(six.b(fake_data), buff_data)