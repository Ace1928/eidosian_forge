from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_areas_filter(filter_dict):
    filter_cmd = 'filter '
    if filter_dict.get('prefix_list'):
        filter_cmd = filter_cmd + filter_dict.get('filter')
    elif filter_dict.get('address'):
        filter_cmd = filter_cmd + filter_dict.get('address')
    else:
        filter_cmd = filter_cmd + filter_dict.get('subnet_address') + ' ' + filter_dict.get('subnet_mask')
    return filter_cmd