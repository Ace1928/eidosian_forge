import collections
import copy
import functools
import itertools
import math
from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import timeutils
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import function
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import support
from heat.scaling import rolling_update
from heat.scaling import template as scl_template
def _resource_names(self, size=None):
    name_skiplist = self._name_skiplist()
    if size is None:
        size = self.get_size()

    def is_skipped(name):
        return name in name_skiplist
    candidates = map(str, itertools.count())
    return itertools.islice(itertools.filterfalse(is_skipped, candidates), size)