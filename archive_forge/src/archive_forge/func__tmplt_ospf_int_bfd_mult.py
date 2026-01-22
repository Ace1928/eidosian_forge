from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_int_bfd_mult(config_data):
    command = _compute_command(config_data)
    bfd = config_data['address_family']['bfd']
    if bfd.get('multiplier'):
        command += ' bfd multiplier ' + str(bfd['multiplier'])
    return command