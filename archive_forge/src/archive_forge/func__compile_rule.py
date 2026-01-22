import configparser
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import policy
import glance.api.policy
from glance.common import exception
from glance.i18n import _, _LE, _LW
def _compile_rule(self, rule):
    try:
        return re.compile(rule)
    except Exception as e:
        msg = _LE('Encountered a malformed property protection rule %(rule)s: %(error)s.') % {'rule': rule, 'error': e}
        LOG.error(msg)
        raise InvalidPropProtectConf()