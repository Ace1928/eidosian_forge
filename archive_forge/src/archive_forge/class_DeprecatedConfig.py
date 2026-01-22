import functools
import inspect
import logging
from oslo_config import cfg
from oslo_log._i18n import _
class DeprecatedConfig(Exception):
    message = _('Fatal call to deprecated config: %(msg)s')

    def __init__(self, msg):
        super(Exception, self).__init__(self.message % dict(msg=msg))