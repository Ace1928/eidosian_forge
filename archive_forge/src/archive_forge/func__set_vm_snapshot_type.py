import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
@_utils.required_vm_version(min_version=constants.VM_VERSION_6_2)
def _set_vm_snapshot_type(self, vmsettings, snapshot_type):
    vmsettings.UserSnapshotType = snapshot_type