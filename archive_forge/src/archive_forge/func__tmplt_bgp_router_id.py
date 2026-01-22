from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_router_id(config_data):
    command = 'router-id {router_id}'.format(**config_data)
    return command