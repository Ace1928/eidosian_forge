from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_set_vswitch_port_vsid_already_set(self, mock_get_elem_assoc_cls):
    self._mock_get_switch_port_alloc()
    mock_sec_settings = mock.MagicMock(AllowMacSpoofing=mock.sentinel.state)
    mock_get_elem_assoc_cls.return_value = (mock_sec_settings, True)
    self.netutils.set_vswitch_port_mac_spoofing(mock.sentinel.switch_port_name, mock.sentinel.state)
    self.assertFalse(self.netutils._jobutils.add_virt_feature.called)