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
def _is_valid_constraint(self, constraint):
    valid_types = getattr(constraint, 'valid_types', [])
    return any((self.type == getattr(self, t, None) for t in valid_types))