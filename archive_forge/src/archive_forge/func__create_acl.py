import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def _create_acl(self, direction, acl_type, action):
    acl = self._create_default_setting_data(self._PORT_ALLOC_ACL_SET_DATA)
    acl.set(Direction=direction, AclType=acl_type, Action=action, Applicability=self._ACL_APPLICABILITY_LOCAL)
    return acl