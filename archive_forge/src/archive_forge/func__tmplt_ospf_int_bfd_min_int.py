from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_int_bfd_min_int(config_data):
    command = _compute_command(config_data)
    bfd = config_data['address_family']['bfd']
    if bfd.get('minimum_interval'):
        command += ' bfd minimum-interval ' + str(bfd['minimum_interval'])
    return command