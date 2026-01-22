import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ResourceTypeUnavailable(HeatException):
    error_code = '99001'