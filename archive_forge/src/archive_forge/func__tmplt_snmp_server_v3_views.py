from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_v3_views(config_data):
    config_data = config_data['snmp_v3']['views']
    command = 'service snmp v3 view {view} oid {oid}'.format(**config_data)
    if 'exclude' in config_data:
        command += ' exclude {exclude}'.format(**config_data)
    if 'mask' in config_data:
        command += ' mask {mask}'.format(**config_data)
    return command