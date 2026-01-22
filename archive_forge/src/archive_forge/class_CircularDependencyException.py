import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class CircularDependencyException(HeatException):
    msg_fmt = _('Circular Dependency Found: %(cycle)s')