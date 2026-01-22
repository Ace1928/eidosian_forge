from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_graceful_restart(config_data):
    command = 'graceful-restart'
    if config_data.get('restart_time'):
        command += ' restart-time {restart_time}'.format(**config_data)
    if config_data.get('stalepath_time'):
        command += ' stalepath-time {stalepath_time}'.format(**config_data)
    return command