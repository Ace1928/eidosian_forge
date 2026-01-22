from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_mpls_ldp(config_data):
    if 'ldp' in config_data['mpls']:
        command = 'mpls ldp'
        if 'autoconfig' in config_data['mpls']['ldp']:
            command += ' autoconfig'
            if 'area' in config_data['mpls']['ldp']['autoconfig']:
                command += ' area {area}'.format(**config_data['mpls']['ldp']['autoconfig'])
        elif 'sync' in config_data['mpls']['ldp']:
            command += ' sync'
    return command