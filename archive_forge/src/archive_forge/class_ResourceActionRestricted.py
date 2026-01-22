import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ResourceActionRestricted(HeatException):
    msg_fmt = _('%(action)s is restricted for resource.')