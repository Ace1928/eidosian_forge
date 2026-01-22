import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class KeystoneServiceNameConflict(HeatException):
    msg_fmt = _('Keystone has more than one service with same name %(service)s. Please use service id instead of name')