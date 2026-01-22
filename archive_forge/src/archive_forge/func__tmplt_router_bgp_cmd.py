from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_router_bgp_cmd(config_data):
    command = 'router bgp {as_number}'.format(**config_data)
    return command