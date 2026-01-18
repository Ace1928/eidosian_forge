from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils10
from os_win.utils import jobutils
def test_set_snapshot_type(self):
    vmsettings = mock.Mock(Version='6.2')
    self._vmutils._set_vm_snapshot_type(vmsettings, mock.sentinel.snapshot_type)
    self.assertEqual(mock.sentinel.snapshot_type, vmsettings.UserSnapshotType)