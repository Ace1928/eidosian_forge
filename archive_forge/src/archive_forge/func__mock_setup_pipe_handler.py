import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
def _mock_setup_pipe_handler(self):
    self._handler._log_file_handle = mock.Mock()
    self._handler._pipe_handle = mock.sentinel.pipe_handle
    self._r_worker = mock.Mock()
    self._w_worker = mock.Mock()
    self._handler._workers = [self._r_worker, self._w_worker]
    self._handler._r_buffer = mock.Mock()
    self._handler._w_buffer = mock.Mock()
    self._handler._r_overlapped = mock.Mock()
    self._handler._w_overlapped = mock.Mock()
    self._handler._r_completion_routine = mock.Mock()
    self._handler._w_completion_routine = mock.Mock()