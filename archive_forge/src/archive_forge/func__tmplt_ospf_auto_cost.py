from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_auto_cost(config_data):
    if 'auto_cost' in config_data:
        command = 'auto-cost'
        if 'reference_bandwidth' in config_data['auto_cost']:
            command += ' reference-bandwidth {reference_bandwidth}'.format(**config_data['auto_cost'])
        return command