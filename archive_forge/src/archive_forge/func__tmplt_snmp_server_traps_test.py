from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_test(config_data):
    command = 'snmp-server enable traps test'
    el = config_data['traps']['test']
    if el.get('arista_test_notification'):
        command += ' arista-test-notification'
    return command