import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class OrphanedObjectError(HeatException):
    msg_fmt = _('Cannot call %(method)s on orphaned %(objtype)s object')