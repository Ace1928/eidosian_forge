from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_discard_route(config_data):
    if 'discard_route' in config_data:
        command = 'discard-route'
        if 'external' in config_data['discard_route']:
            command += ' external {external}'.format(**config_data['discard_route'])
        if 'internal' in config_data['discard_route']:
            command += ' internal {internal}'.format(**config_data['discard_route'])
        return command