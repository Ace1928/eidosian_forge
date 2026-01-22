from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_vrf_local_rib_criteria(config_data):
    if 'local_rib_criteria' in config_data:
        command = 'local-rib-criteria'
        if 'forwarding_address' in config_data['local_rib_criteria']:
            command += ' forwarding-address'
        if 'inter_area_summary' in config_data['local_rib_criteria']:
            command += ' inter-area-summary'
        if 'nssa_translation' in config_data['local_rib_criteria']:
            command += ' nssa-translation'
        return command