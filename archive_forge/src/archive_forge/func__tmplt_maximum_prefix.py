from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_maximum_prefix(data):
    data = data['maximum_prefix']
    cmd = 'maximum-prefix {max_prefix_limit}'.format(**data)
    if 'generate_warning_threshold' in data:
        cmd += ' {generate_warning_threshold}'.format(**data)
    if 'restart_interval' in data:
        cmd += ' restart {restart_interval}'.format(**data)
    if data.get('warning_only'):
        cmd += ' warning-only'
    return cmd