from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_logging_global_vrf_hosts(config_data):
    el = config_data['vrfs']
    command = 'logging vrf ' + el['name'] + ' host '
    el = el['hosts']
    command += el['name']
    if el.get('add'):
        command += ' add'
    if el.get('remove'):
        command += ' remove'
    if el.get('port'):
        command += ' ' + str(el['port'])
    if el.get('protocol'):
        command += ' protocol ' + el['protocol']
    return command