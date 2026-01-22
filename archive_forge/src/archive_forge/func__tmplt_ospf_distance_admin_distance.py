from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_distance_admin_distance(config_data):
    if 'admin_distance' in config_data['distance']:
        command = 'distance {distance}'.format(**config_data['distance']['admin_distance'])
        if 'address' in config_data['distance']['admin_distance']:
            command += ' {address} {wildcard_bits}'.format(**config_data['distance']['admin_distance'])
        if 'acl' in config_data['distance']['admin_distance']:
            command += ' {acl}'.format(**config_data['distance']['admin_distance'])
        return command