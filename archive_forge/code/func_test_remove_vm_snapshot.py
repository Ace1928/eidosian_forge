from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_remove_vm_snapshot(self):
    mock_svc = self._get_snapshot_service()
    getattr(mock_svc, self._DESTROY_SNAPSHOT).return_value = (self._FAKE_JOB_PATH, self._FAKE_RET_VAL)
    self._vmutils.remove_vm_snapshot(self._FAKE_SNAPSHOT_PATH)
    getattr(mock_svc, self._DESTROY_SNAPSHOT).assert_called_with(self._FAKE_SNAPSHOT_PATH)