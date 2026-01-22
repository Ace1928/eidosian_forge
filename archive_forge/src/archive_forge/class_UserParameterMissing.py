import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class UserParameterMissing(HeatException):
    msg_fmt = _('The Parameter (%(key)s) was not provided.')