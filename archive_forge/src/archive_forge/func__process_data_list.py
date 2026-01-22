import sys
import re
import copy
import logging
from pyomo.common.log import is_debug_set
from pyomo.common.collections import Bunch, OrderedDict
from pyomo.common.errors import ApplicationError
from pyomo.dataportal.parse_datacmds import parse_data_commands, _re_number
from pyomo.dataportal.factory import DataManagerFactory, UnknownDataManager
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.util import flatten_tuple
def _process_data_list(param_name, dim, cmd):
    """ Called by _process_param() to process a list of data for a Parameter.
 """
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug('process_data_list %d %s', dim, cmd)
    if len(cmd) % (dim + 1) != 0:
        msg = "Parameter '%s' defined with '%d' dimensions, but data has '%d' values: %s."
        msg = msg % (param_name, dim, len(cmd), cmd)
        if len(cmd) % (dim + 1) == dim:
            msg += ' Are you missing a value for a %d-dimensional index?' % dim
        elif len(cmd) % dim == 0:
            msg += ' Are you missing the values for %d-dimensional indices?' % dim
        else:
            msg += ' Data declaration must be given in multiples of %d.' % (dim + 1)
        raise ValueError(msg)
    ans = {}
    if dim == 0:
        ans[None] = cmd[0]
        return ans
    i = 0
    while i < len(cmd):
        if dim > 1:
            ndx = tuple(cmd[i:i + dim])
        else:
            ndx = cmd[i]
        if cmd[i + dim] != '.':
            ans[ndx] = cmd[i + dim]
        i += dim + 1
    return ans