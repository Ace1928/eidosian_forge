from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_host_del(verb):
    cmd = 'logging host'
    if verb.get('host'):
        cmd += ' {hostname}'.format(hostname=verb['host'])
    if verb.get('ipv6'):
        cmd += ' ipv6 {ipv6}'.format(ipv6=verb['ipv6'])
    if verb.get('transport'):
        cmd = None
    return cmd