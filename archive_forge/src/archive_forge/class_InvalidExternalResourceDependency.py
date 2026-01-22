import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidExternalResourceDependency(HeatException):
    msg_fmt = _('Invalid dependency with external %(resource_type)s resource: %(external_id)s')