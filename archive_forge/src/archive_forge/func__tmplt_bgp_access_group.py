from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_access_group(config_data):
    if config_data.get('afi') == 'ipv4':
        afi = 'ip'
    else:
        afi = 'ipv6'
    command = afi + ' access-group {acl_name}'.format(**config_data)
    if config_data.get('direction'):
        command += ' {direction}'.format(**config_data)
    return command