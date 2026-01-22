from __future__ import absolute_import, division, print_function
from natsort import (
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.interfaces_util import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import re
import traceback
def __derive_interface_config_delete_op(key_set, command, exist_conf):
    new_conf = exist_conf
    intf_name = command['name']
    for attr in eth_attribute:
        if attr in command:
            if attr == 'speed':
                new_conf[attr] = default_intf_speeds[intf_name]
            elif attr == 'advertised_speed':
                if new_conf[attr] is not None:
                    new_conf[attr] = list(set(new_conf[attr]).difference(command[attr]))
                    if new_conf[attr] == []:
                        new_conf[attr] = None
            elif attr == 'auto_negotiate':
                new_conf[attr] = False
                if new_conf.get('advertised_speed') is not None:
                    new_conf['advertised_speed'] = None
            else:
                new_conf[attr] = attributes_default_value[attr]
    return (True, new_conf)