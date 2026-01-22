from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospfv3_area_authentication_ipsec(config_data):
    if 'authentication' in config_data:
        command = 'area {area_id} '.format(**config_data)
        if config_data['authentication'].get('ipsec'):
            command += 'authentication ipsec'
            md = config_data['authentication'].get('ipsec')
            if md.get('spi'):
                command += ' spi ' + str(md.get('spi'))
            if md.get('algorithim_type'):
                command += ' ' + md.get('algorithim_type')
            if md.get('clear_key'):
                command += ' clear ' + md.get('clear_key')
            elif md.get('password_key'):
                command += ' password ' + md.get('password_key')
            elif md.get('key'):
                command += ' ' + md.get('key')
            return command