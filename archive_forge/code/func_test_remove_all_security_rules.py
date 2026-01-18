from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_remove_all_security_rules(self, mock_get_elem_assoc_cls):
    mock_acl = self._setup_security_rule_test(mock_get_elem_assoc_cls)[1]
    self.netutils.remove_all_security_rules(self._FAKE_PORT_NAME)
    mock_remove_features = self.netutils._jobutils.remove_multiple_virt_features
    mock_remove_features.assert_called_once_with([mock_acl])