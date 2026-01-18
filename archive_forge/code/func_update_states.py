from __future__ import absolute_import, division, print_function
import re
import json
import ast
from copy import copy
from itertools import (count, groupby)
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.common.network import (
from ansible.module_utils.common.validation import check_required_arguments
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def update_states(commands, state):
    ret_list = list()
    if commands:
        if isinstance(commands, list):
            for command in commands:
                ret = command.copy()
                ret.update({'state': state})
                ret_list.append(ret)
        elif isinstance(commands, dict):
            ret_list.append(commands.copy())
            ret_list[0].update({'state': state})
    return ret_list