from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_timers_lsa(config_data):
    command = ''
    if 'lsa' in config_data['timers']:
        command += 'timers lsa {direction}'.format(**config_data['timers']['lsa'])
        if config_data['timers']['lsa']['direction'] == 'rx':
            command += ' min interval '
        else:
            command += ' delay initial '
        if config_data['timers']['lsa'].get('initial'):
            command += str(config_data['timers']['lsa']['initial'])
        if config_data['timers']['lsa'].get('min'):
            command += str(config_data['timers']['lsa']['min'])
        if config_data['timers']['lsa'].get('max'):
            command += str(config_data['timers']['lsa']['max'])
    return command