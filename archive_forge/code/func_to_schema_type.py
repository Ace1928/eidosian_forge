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
def to_schema_type(self, value):
    """Returns the value in the schema's data type."""
    try:
        if self.type == self.INTEGER:
            num = Schema.str_to_num(value)
            if isinstance(num, float):
                raise ValueError(_('%s is not an integer.') % num)
            return num
        elif self.type == self.NUMBER:
            return Schema.str_to_num(value)
        elif self.type == self.STRING:
            return str(value)
        elif self.type == self.BOOLEAN:
            return strutils.bool_from_string(str(value), strict=True)
    except ValueError:
        raise ValueError(_('Value "%(val)s" is invalid for data type "%(type)s".') % {'val': value, 'type': self.type})
    return value