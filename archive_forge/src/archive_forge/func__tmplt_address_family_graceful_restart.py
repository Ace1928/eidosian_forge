from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_address_family_graceful_restart(config_data):
    if 'graceful_restart' in config_data:
        command = 'graceful_restart {enable}'.format(**config_data['graceful_restart'])
        if 'disable' in config_data['graceful_restart']:
            command += ' disable'
        elif 'strict_lsa_checking' in config_data['graceful_restart']:
            command += ' strict-lsa-checking'
        return command