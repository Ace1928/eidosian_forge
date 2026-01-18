import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
def test_start_io_worker_with_buffer_update_method(self):
    self._test_start_io_worker(buff_update_func=mock.Mock())