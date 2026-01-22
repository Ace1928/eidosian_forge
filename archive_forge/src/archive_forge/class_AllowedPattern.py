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
class AllowedPattern(Constraint):
    """Constrain values to a predefined regular expression pattern.

    Serializes to JSON as::

        {
            'allowed_pattern': <pattern>,
            'description': <description>
        }
    """
    valid_types = (Schema.STRING_TYPE,)

    def __init__(self, pattern, description=None):
        super(AllowedPattern, self).__init__(description)
        if not isinstance(pattern, str):
            raise exception.InvalidSchemaError(message=_('AllowedPattern must be a string'))
        self.pattern = pattern
        self.match = re.compile(pattern).match

    def _str(self):
        return _('Value must match pattern: %s') % self.pattern

    def _err_msg(self, value):
        return _('"%(value)s" does not match pattern "%(pattern)s"') % {'value': value, 'pattern': self.pattern}

    def _is_valid(self, value, schema, context):
        match = self.match(value)
        return match is not None and match.end() == len(value)

    def _constraint(self):
        return self.pattern