from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_adjacency_stagger(config_data):
    if 'adjacency_stagger' in config_data:
        command = 'adjacency stagger'.format(**config_data)
        if config_data['adjacency_stagger'].get('min_adjacency') and config_data['adjacency_stagger'].get('min_adjacency'):
            command += ' {0} {1}'.format(config_data['adjacency_stagger'].get('min_adjacency'), config_data['adjacency_stagger'].get('max_adjacency'))
        elif config_data['adjacency_stagger'].get('disable'):
            command += ' disable'
        return command