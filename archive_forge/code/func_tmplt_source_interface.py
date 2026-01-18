from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_source_interface(verb):
    cmd = 'logging source-interface'
    if verb.get('interface'):
        cmd += ' {interface}'.format(interface=verb['interface'])
    if verb.get('vrf'):
        cmd += ' vrf {vrf}'.format(vrf=verb['vrf'])
    return cmd