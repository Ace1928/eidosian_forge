from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_is_port_vm_started')
def test_test_is_metrics_collection_allowed_false(self, mock_is_started):
    self._test_is_metrics_collection_allowed(mock_vm_started=mock_is_started, acls=[], expected_result=False)