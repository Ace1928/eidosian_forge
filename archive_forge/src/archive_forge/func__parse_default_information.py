from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_default_information(default_dict):
    def_cmd = 'default-information originate'
    for def_key in sorted(default_dict.keys()):
        if def_key == 'always':
            if default_dict.get(def_key):
                def_cmd = def_cmd + ' ' + def_key
        elif def_key in ['metric', 'metric_type', 'route_map']:
            if default_dict.get(def_key):
                k = re.sub('_', '-', def_key)
                def_cmd = def_cmd + ' ' + k + ' ' + str(default_dict[def_key])
    return def_cmd