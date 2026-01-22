from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_vrf_cmd(process):
    command = 'router ospfv3'
    vrf = '{vrf}'.format(**process)
    if 'vrf' in process and vrf != 'default':
        command += ' vrf ' + vrf
    return command