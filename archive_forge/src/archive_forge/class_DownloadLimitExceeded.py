import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class DownloadLimitExceeded(HeatException):
    msg_fmt = _('Permissible download limit exceeded: %(message)s')