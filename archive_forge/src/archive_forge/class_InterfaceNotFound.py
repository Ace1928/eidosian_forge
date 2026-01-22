from heat.common import exception
from heat.common.i18n import _
class InterfaceNotFound(exception.HeatException):
    msg_fmt = _('No network interface found for server %(id)s.')