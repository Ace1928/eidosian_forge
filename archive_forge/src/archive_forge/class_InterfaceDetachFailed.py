import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InterfaceDetachFailed(HeatException):
    msg_fmt = _('Failed to detach interface (%(port)s) from server (%(server)s)')