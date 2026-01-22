from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_protocol(config_data):
    if 'protocol_shutdown' in config_data:
        command = 'protocol'
        if 'set' in config_data['protocol_shutdown']:
            command += ' shutdown'
        elif config_data['shutdown'].get('host_mode'):
            command += ' shutdown host-mode'
        elif config_data['shutdown'].get('on_reload'):
            command += ' shutdown on-reload'
        return command