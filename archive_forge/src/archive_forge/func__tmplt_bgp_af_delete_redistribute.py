from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_af_delete_redistribute(config_data):
    afi = config_data['address_family']['afi'] + '-unicast'
    command = 'protocols bgp {as_number} address-family '.format(**config_data)
    config_data = config_data['address_family']
    command += afi + ' redistribute {protocol}'.format(**config_data['redistribute'])
    return command