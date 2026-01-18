import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe.NamedPipeHandler, 'stop')
@mock.patch.object(namedpipe.NamedPipeHandler, '_open_pipe')
def test_start_pipe_handler_exception(self, mock_open_pipe, mock_stop_handler):
    mock_open_pipe.side_effect = Exception
    self.assertRaises(exceptions.OSWinException, self._handler.start)
    mock_stop_handler.assert_called_once_with()