import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidServiceVersion(HeatException):
    msg_fmt = _('Invalid service %(service)s version %(version)s')