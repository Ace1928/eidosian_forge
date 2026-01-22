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
def _filter_acls(self, acls, action, direction, acl_type, remote_addr=''):
    return [v for v in acls if v.Action == action and v.Direction == direction and (v.AclType == acl_type) and (v.RemoteAddress == remote_addr)]