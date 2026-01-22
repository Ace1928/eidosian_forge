import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe.NamedPipeHandler, '_cleanup_handles')
def _test_start_io_worker(self, mock_cleanup_handles, buff_update_func=None, exception=None):
    self._handler._stopped.isSet.side_effect = [False, True]
    self._handler._pipe_handle = mock.sentinel.pipe_handle
    self._handler.stop = mock.Mock()
    io_func = mock.Mock(side_effect=exception)
    fake_buffer = 'fake_buffer'
    self._handler._start_io_worker(io_func, fake_buffer, mock.sentinel.overlapped_structure, mock.sentinel.completion_routine, buff_update_func)
    if buff_update_func:
        num_bytes = buff_update_func()
    else:
        num_bytes = len(fake_buffer)
    io_func.assert_called_once_with(mock.sentinel.pipe_handle, fake_buffer, num_bytes, mock.sentinel.overlapped_structure, mock.sentinel.completion_routine)
    if exception:
        self._handler._stopped.set.assert_called_once_with()
    mock_cleanup_handles.assert_called_once_with()