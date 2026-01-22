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
def _try_rolling_update(self):
    if self.update_policy[self.ROLLING_UPDATE]:
        policy = self.update_policy[self.ROLLING_UPDATE]
        return self._replace(policy[self.MIN_IN_SERVICE], policy[self.MAX_BATCH_SIZE], policy[self.PAUSE_TIME])