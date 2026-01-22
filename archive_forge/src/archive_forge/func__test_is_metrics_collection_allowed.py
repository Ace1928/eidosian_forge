from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def _test_is_metrics_collection_allowed(self, mock_get_elem_assoc_cls, mock_vm_started, acls, expected_result):
    mock_port = self._mock_get_switch_port_alloc()
    mock_acl = mock.MagicMock()
    mock_acl.Action = self.netutils._ACL_ACTION_METER
    mock_get_elem_assoc_cls.return_value = acls
    mock_vm_started.return_value = True
    result = self.netutils.is_metrics_collection_allowed(self._FAKE_PORT_NAME)
    self.assertEqual(expected_result, result)
    mock_get_elem_assoc_cls.assert_called_once_with(self.netutils._conn, self.netutils._PORT_ALLOC_ACL_SET_DATA, element_instance_id=mock_port.InstanceID)