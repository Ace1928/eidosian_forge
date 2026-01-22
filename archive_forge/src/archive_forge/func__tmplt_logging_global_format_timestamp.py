from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_logging_global_format_timestamp(config_data):
    command = ''
    el = config_data['format']['timestamp']
    if el.get('traditional'):
        command = 'logging format timestamp traditional'
        if el['traditional'].get('year'):
            if el['traditional']['year']:
                command += ' year'
        if el['traditional'].get('timezone'):
            if el['traditional']['timezone']:
                command += ' timezone'
    return command