from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _compute_src_dest(dir_dict):
    cmd = ''
    if 'any' in dir_dict:
        cmd += 'any '
    elif 'host' in dir_dict:
        cmd += 'host {0} '.format(dir_dict['host'])
    elif 'net_group' in dir_dict:
        cmd += 'net-group {0} '.format(dir_dict['net_group'])
    elif 'port_group' in dir_dict:
        cmd += 'port-group {0} '.format(dir_dict['port_group'])
    elif 'prefix' in dir_dict:
        cmd += '{0} '.format(dir_dict['prefix'])
    else:
        cmd += '{0} {1} '.format(dir_dict['address'], dir_dict['wildcard_bits'])
    if 'port_protocol' in dir_dict:
        protocol_range = dir_dict['port_protocol'].get('range')
        if protocol_range:
            cmd += 'range {0} {1} '.format(protocol_range['start'], protocol_range['end'])
        else:
            for key, value in iteritems(dir_dict['port_protocol']):
                cmd += '{0} {1} '.format(key, value)
    return cmd