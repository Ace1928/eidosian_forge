from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_ranges(config_data):
    if 'ranges' in config_data:
        commands = []
        for k, v in iteritems(config_data['ranges']):
            cmd = 'area {area_id} range'.format(**config_data)
            temp_cmd = ' {address} {netmask}'.format(**v)
            if 'advertise' in v:
                temp_cmd += ' advertise'
            elif 'not_advertise' in v:
                temp_cmd += ' not-advertise'
            if 'cost' in v:
                temp_cmd += ' cost {cost}'.format(**v)
            cmd += temp_cmd
            commands.append(cmd)
        return commands