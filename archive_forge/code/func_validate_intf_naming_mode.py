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
def validate_intf_naming_mode(intf_name, module):
    global intf_naming_mode
    if intf_naming_mode == '':
        intf_naming_mode = get_device_interface_naming_mode(module)
    if intf_naming_mode != '':
        ansible_intf_naming_mode = find_intf_naming_mode(intf_name)
        if intf_naming_mode != ansible_intf_naming_mode:
            err = 'Interface naming mode configured on switch {naming_mode}, {intf_name} is not valid'.format(naming_mode=intf_naming_mode, intf_name=intf_name)
            module.fail_json(msg=err, code=400)