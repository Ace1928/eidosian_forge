from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_set_vm_serial_port_conn(self):
    self._lookup_vm()
    mock_com_1 = mock.Mock()
    mock_com_2 = mock.Mock()
    self._vmutils._get_vm_serial_ports = mock.Mock(return_value=[mock_com_1, mock_com_2])
    self._vmutils.set_vm_serial_port_connection(mock.sentinel.vm_name, port_number=1, pipe_path=mock.sentinel.pipe_path)
    self.assertEqual([mock.sentinel.pipe_path], mock_com_1.Connection)
    self._vmutils._jobutils.modify_virt_resource.assert_called_once_with(mock_com_1)