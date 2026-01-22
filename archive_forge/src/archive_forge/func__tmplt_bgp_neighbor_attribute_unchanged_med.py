from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_neighbor_attribute_unchanged_med(config_data):
    command = 'protocols bgp {as_number} '.format(**config_data) + 'neighbor {address} attribute-unchanged med'.format(**config_data['neighbor'])
    return command