from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_filter(verb):
    cmd = 'logging filter'
    if verb.get('url'):
        cmd += ' {url}'.format(url=verb['url'])
    if verb.get('order'):
        cmd += ' {order}'.format(order=verb['order'])
    if verb.get('args'):
        cmd += ' args {args}'.format(args=verb['args'])
    return cmd