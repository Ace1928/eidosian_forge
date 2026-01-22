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
def _process_set(cmd, _model, _data):
    """
    Called by _process_data() to process a set declaration.
    """
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug('DEBUG: _process_set(start) %s', cmd)
    if type(cmd[2]) is tuple:
        ndx = cmd[2]
        if len(ndx) == 0:
            raise ValueError('Illegal indexed set specification encountered: ' + str(cmd[1]))
        elif len(ndx) == 1:
            ndx = ndx[0]
        if cmd[1] not in _data:
            _data[cmd[1]] = {}
        _data[cmd[1]][ndx] = _process_set_data(cmd[4:], cmd[1], _model)
    elif cmd[2] == ':':
        _data[cmd[1]] = {}
        _data[cmd[1]][None] = []
        i = 3
        while cmd[i] != ':=':
            i += 1
        ndx1 = cmd[3:i]
        i += 1
        while i < len(cmd):
            ndx = cmd[i]
            for j in range(0, len(ndx1)):
                if cmd[i + j + 1] == '+':
                    _data[cmd[1]][None].append((ndx1[j], cmd[i]))
            i += len(ndx1) + 1
    else:
        _data[cmd[1]] = {}
        _data[cmd[1]][None] = _process_set_data(cmd[3:], cmd[1], _model)