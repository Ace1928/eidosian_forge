from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_label_mode(conf):
    if 'label_mode' in conf:
        command = 'vrf all label mode'
        if 'per_ce' in conf.get('label_mode'):
            command += ' per-ce'
        elif 'per_vrf' in conf.get('label_mode'):
            command += ' per-vrf'
        elif 'per_prefix' in conf.get('label_mode'):
            command += ' per-prefix'
        elif 'route_policy' in conf.get('label_mode'):
            command += ' route-policy ' + conf['route_policy']
    return command