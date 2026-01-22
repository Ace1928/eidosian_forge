from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_logging_trap(config_data):
    command = 'logging trap'
    if 'severity' in config_data['trap']:
        command += ' ' + config_data['trap']['severity']
    return command