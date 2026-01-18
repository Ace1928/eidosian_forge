from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
@staticmethod
def yaml_bool_to_python_bool(yaml_bool):
    """Convert the input YAML bool value to a Python bool value"""
    boolval = False
    if yaml_bool is None:
        boolval = False
    elif yaml_bool:
        boolval = True
    return boolval