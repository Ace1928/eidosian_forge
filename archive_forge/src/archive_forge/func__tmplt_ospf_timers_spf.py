from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_timers_spf(config_data):
    command = ''
    if 'spf' in config_data['timers']:
        command += 'timers spf delay initial '
        command += '{initial} {min} {max}'.format(**config_data['timers']['spf'])
    return command