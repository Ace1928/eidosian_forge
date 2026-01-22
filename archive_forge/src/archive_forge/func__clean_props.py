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
def _clean_props(self, res_defn):
    res_def = copy.deepcopy(res_defn)
    props = res_def.get(self.RESOURCE_DEF_PROPERTIES)
    if props:
        clean = dict(((k, v) for k, v in props.items() if v is not None))
        props = clean
        res_def[self.RESOURCE_DEF_PROPERTIES] = props
    return res_def