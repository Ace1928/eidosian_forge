from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_max_lsa(config_data):
    if 'max_lsa' in config_data:
        command = 'max-lsa {number}'.format(**config_data['max_lsa'])
        if 'threshold_value' in config_data['max_lsa']:
            command += ' {threshold_value}'.format(**config_data['max_lsa'])
        if 'ignore_count' in config_data['max_lsa']:
            command += ' ignore-count {ignore_count}'.format(**config_data['max_lsa'])
        if 'ignore_time' in config_data['max_lsa']:
            command += ' ignore-time {ignore_time}'.format(**config_data['max_lsa'])
        if 'reset_time' in config_data['max_lsa']:
            command += ' reset-time {reset_time}'.format(**config_data['max_lsa'])
        if 'warning_only' in config_data['max_lsa']:
            command += ' warning-only'
        return command