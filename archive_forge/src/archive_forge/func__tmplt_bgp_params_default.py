from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_params_default(config_data):
    command = 'protocols bgp {as_number} parameters default'.format(**config_data)
    if config_data['bgp_params']['default'].get('no_ipv4_unicast'):
        command += ' no-ipv4-unicast'
    if config_data['bgp_params']['default'].get('local_pref'):
        command += ' local-pref {local_pref}'.format(**config_data['bgp_params']['default'])
    return command