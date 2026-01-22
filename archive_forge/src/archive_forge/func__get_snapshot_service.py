from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def _get_snapshot_service(self):
    return self._vmutils._conn.Msvm_VirtualSystemSnapshotService()[0]