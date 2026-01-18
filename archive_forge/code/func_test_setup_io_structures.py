import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
def test_setup_io_structures(self):
    self._handler._setup_io_structures()
    self.assertEqual(self._ioutils.get_buffer.return_value, self._handler._r_buffer)
    self.assertEqual(self._ioutils.get_buffer.return_value, self._handler._w_buffer)
    self.assertEqual(self._ioutils.get_new_overlapped_structure.return_value, self._handler._r_overlapped)
    self.assertEqual(self._ioutils.get_new_overlapped_structure.return_value, self._handler._w_overlapped)
    self.assertEqual(self._ioutils.get_completion_routine.return_value, self._handler._r_completion_routine)
    self.assertEqual(self._ioutils.get_completion_routine.return_value, self._handler._w_completion_routine)
    self.assertIsNone(self._handler._log_file_handle)
    self._ioutils.get_buffer.assert_has_calls([mock.call(constants.SERIAL_CONSOLE_BUFFER_SIZE)] * 2)
    self._ioutils.get_completion_routine.assert_has_calls([mock.call(self._handler._read_callback), mock.call()])