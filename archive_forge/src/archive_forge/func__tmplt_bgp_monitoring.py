from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_monitoring(config_data):
    cmd = 'monitoring'
    command = ''
    if config_data.get('timestamp'):
        command = cmd + ' timestamp {timestamp}'.format(**config_data)
    if config_data.get('port'):
        command = cmd + ' port {port}'.format(**config_data)
    if config_data.get('received'):
        command = cmd + ' received routes {received}'.format(**config_data)
    if config_data.get('station'):
        command = cmd + ' station {station}'.format(**config_data)
    return command