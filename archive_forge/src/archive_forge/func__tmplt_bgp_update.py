from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_update(config_data):
    command = 'update {wait_for}'.format(**config_data['update'])
    if config_data['update'].get('batch_size'):
        command += ' {batch_size}'.format(**config_data['update'])
    return command