import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidTemplateReference(HeatException):
    msg_fmt = _('The specified reference "%(resource)s" (in %(key)s) is incorrect.')