from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_network(config_data):
    if 'network' in config_data:
        command = 'network'
        if 'address' in config_data['network']:
            command += ' {address} {wildcard_bits}'.format(**config_data['network'])
        if 'area' in config_data['network']:
            command += ' area {area}'.format(**config_data['network'])
        return command