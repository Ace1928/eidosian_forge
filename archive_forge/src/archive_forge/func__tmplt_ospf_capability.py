from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_capability(config_data):
    if 'capability' in config_data:
        if 'lls' in config_data['capability']:
            command = 'capability lls'
        elif 'opaque' in config_data['capability']:
            command = 'capability opaque'
        elif 'transit' in config_data['capability']:
            command = 'capability transit'
        elif 'vrf_lite' in config_data['capability']:
            command = 'capability vrf-lite'
        return command