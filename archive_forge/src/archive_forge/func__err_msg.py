import collections
import json
import numbers
import re
from oslo_cache import core
from oslo_config import cfg
from oslo_log import log
from oslo_utils import reflection
from oslo_utils import strutils
from heat.common import cache
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resources
def _err_msg(self, value):
    constraint = self.custom_constraint
    if constraint is None:
        return _('"%(value)s" does not validate %(name)s (constraint not found)') % {'value': value, 'name': self.name}
    error = getattr(constraint, 'error', None)
    if error:
        return error(value)
    return _('"%(value)s" does not validate %(name)s') % {'value': value, 'name': self.name}