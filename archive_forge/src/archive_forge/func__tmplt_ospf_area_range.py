from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_range(config_data):
    if 'area_id' in config_data:
        command = 'area {area_id} range'.format(**config_data)
        if 'address' in config_data:
            command += ' {address}'.format(**config_data)
        if 'subnet_address' in config_data:
            command += ' {subnet_address}'.format(**config_data)
        if 'subnet_mask' in config_data:
            command += ' {subnet_mask}'.format(**config_data)
        if 'advertise' in config_data:
            if config_data.get('advertise'):
                command += ' advertise'
            else:
                command += ' not-advertise'
        if 'cost' in config_data:
            command += ' cost {cost}'.format(**config_data)
        return command