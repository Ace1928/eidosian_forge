import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ResourcePropertyDependency(HeatException):
    msg_fmt = _('%(prop1)s cannot be specified without %(prop2)s.')