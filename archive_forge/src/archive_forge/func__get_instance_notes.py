import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def _get_instance_notes(self, vm_name):
    vmsettings = self._lookup_vm_check(vm_name)
    vm_notes = vmsettings.Notes or []
    return [note for note in vm_notes if note]