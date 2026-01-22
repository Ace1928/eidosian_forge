from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_set_extcomm_rt(data):
    cmd = 'set extcommunity rt'
    extcomm_numbers = ' '.join(data.get('extcommunity_numbers', []))
    if extcomm_numbers:
        cmd += ' ' + extcomm_numbers
    if data.get('additive'):
        cmd += ' additive'
    return cmd