import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def _build_parameter_constraints(res, param):
    constraint_description = []
    for c in param.schema.constraints:
        if isinstance(c, constr.Length):
            if c.min is not None:
                res[rpc_api.PARAM_MIN_LENGTH] = c.min
            if c.max is not None:
                res[rpc_api.PARAM_MAX_LENGTH] = c.max
        elif isinstance(c, constr.Range):
            if c.min is not None:
                res[rpc_api.PARAM_MIN_VALUE] = c.min
            if c.max is not None:
                res[rpc_api.PARAM_MAX_VALUE] = c.max
        elif isinstance(c, constr.Modulo):
            if c.step is not None:
                res[rpc_api.PARAM_STEP] = c.step
            if c.offset is not None:
                res[rpc_api.PARAM_OFFSET] = c.offset
        elif isinstance(c, constr.AllowedValues):
            res[rpc_api.PARAM_ALLOWED_VALUES] = list(c.allowed)
        elif isinstance(c, constr.AllowedPattern):
            res[rpc_api.PARAM_ALLOWED_PATTERN] = c.pattern
        elif isinstance(c, constr.CustomConstraint):
            res[rpc_api.PARAM_CUSTOM_CONSTRAINT] = c.name
        if c.description:
            constraint_description.append(c.description)
    if constraint_description:
        res[rpc_api.PARAM_CONSTRAINT_DESCRIPTION] = ' '.join(constraint_description)