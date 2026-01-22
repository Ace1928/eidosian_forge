from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_params_confederation(config_data):
    command = []
    for list_el in config_data['bgp_params']['confederation']:
        for k, v in iteritems(list_el):
            command.append('protocols bgp {as_number} parameters confederation '.format(**config_data) + k + ' ' + str(v))
    return command