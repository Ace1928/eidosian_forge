from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_int_bfd(config_data):
    if config_data['afi'] == 'ipv4':
        command = 'ip ospf bfd'
    else:
        command = 'ospfv3 bfd'
    return command