from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _tmplt_ospf_int_dead_interval(config_data):
    int_type = get_interface_type(config_data['name'])
    params = _get_parameters(config_data['address_family'])
    command = 'interfaces ' + int_type + ' {name} '.format(**config_data) + params[1] + ' ' + params[0] + ' dead-interval {dead_interval}'.format(**config_data['address_family'])
    return command