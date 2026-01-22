import configparser
import logging
import logging.config
import logging.handlers
import os
import platform
import sys
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_utils import units
from oslo_log._i18n import _
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
def _find_facility(facility):
    valid_facilities = [f for f in ['LOG_KERN', 'LOG_USER', 'LOG_MAIL', 'LOG_DAEMON', 'LOG_AUTH', 'LOG_SYSLOG', 'LOG_LPR', 'LOG_NEWS', 'LOG_UUCP', 'LOG_CRON', 'LOG_AUTHPRIV', 'LOG_FTP', 'LOG_LOCAL0', 'LOG_LOCAL1', 'LOG_LOCAL2', 'LOG_LOCAL3', 'LOG_LOCAL4', 'LOG_LOCAL5', 'LOG_LOCAL6', 'LOG_LOCAL7'] if getattr(syslog, f, None)]
    facility = facility.upper()
    if not facility.startswith('LOG_'):
        facility = 'LOG_' + facility
    if facility not in valid_facilities:
        raise TypeError(_('syslog facility must be one of: %s') % ', '.join(("'%s'" % fac for fac in valid_facilities)))
    return getattr(syslog, facility)