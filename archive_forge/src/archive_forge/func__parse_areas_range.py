from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_areas_range(range_dict):
    range_cmd = ' range '
    if range_dict.get('address'):
        range_cmd = range_cmd + range_dict['address']
    if range_dict.get('subnet_address'):
        range_cmd = range_cmd + range_dict['subnet_address'] + ' ' + range_dict['subnet_mask']
    if range_dict.get('advertise') is not None:
        if range_dict['advertise']:
            range_cmd = range_cmd + ' advertise '
        else:
            range_cmd = range_cmd + ' not-advertise '
    if range_dict.get('cost'):
        range_cmd = range_cmd + ' cost ' + str(range_dict['cost'])
    return range_cmd