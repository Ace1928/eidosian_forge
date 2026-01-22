import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ObjectFieldInvalid(HeatException):
    msg_fmt = _('Field %(field)s of %(objname)s is not an instance of Field')