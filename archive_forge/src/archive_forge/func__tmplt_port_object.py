from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_port_object(config_data):
    if config_data.get('port_object'):
        cmd = 'port-object'
        if config_data['port_object'].get('range'):
            cmd += ' range {start} {end}'.format(**config_data['port_object']['range'])
        else:
            key = list(config_data['port_object'])[0]
            cmd += ' {0} {1}'.format(key, config_data['port_object'][key])
        return cmd