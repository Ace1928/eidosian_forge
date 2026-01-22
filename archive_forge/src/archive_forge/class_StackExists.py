import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class StackExists(HeatException):
    msg_fmt = _('The Stack (%(stack_name)s) already exists.')