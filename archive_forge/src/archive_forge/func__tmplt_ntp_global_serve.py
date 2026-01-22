from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.utils.utils import (
def _tmplt_ntp_global_serve(config_data):
    el = config_data['serve']
    command = 'ntp serve'
    if el.get('access_lists'):
        command += ' {afi} access-group'.format(**el['access_lists'])
        if 'acls' in el['access_lists']:
            command += ' {acl_name} '.format(**el['access_lists']['acls'])
            if el['access_lists']['acls'].get('vrf'):
                command += ' vrf {vrf} '.format(**el['access_lists']['acls'])
            command += el['access_lists']['acls']['direction']
    return command