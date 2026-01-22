from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_maximum_prefix(config_data):
    conf = config_data.get('maximum_prefix', {})
    if conf:
        command = 'maximum-prefix'
        if 'max_limit' in conf:
            command += ' ' + str(conf['max_limit'])
        if 'threshold_value' in conf:
            command += ' ' + str(conf['threshold_value'])
        if 'restart' in conf:
            command += ' restart ' + str(conf['restart'])
        elif 'warning_only' in conf:
            command += ' warning-only'
        elif 'discard_extra_paths' in conf:
            command += ' discard-extra-paths'
    return command