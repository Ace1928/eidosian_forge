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
@property
def _sec_svc(self):
    if not self._sec_svc_attr:
        self._sec_svc_attr = self._conn.Msvm_SecurityService()[0]
    return self._sec_svc_attr