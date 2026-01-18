import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe.NamedPipeHandler, '_start_io_worker')
def test_read_from_pipe(self, mock_start_worker):
    self._mock_setup_pipe_handler()
    self._handler._read_from_pipe()
    mock_start_worker.assert_called_once_with(self._ioutils.read, self._handler._r_buffer, self._handler._r_overlapped, self._handler._r_completion_routine)