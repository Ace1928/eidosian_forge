from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_nsf_ietf(config_data):
    if 'ietf' in config_data['nsf']:
        command = 'nsf ietf helper'
        if 'disable' in config_data['nsf']['ietf']:
            command += ' disable'
        elif 'strict_lsa_checking' in config_data['nsf']['ietf']:
            command += ' strict-lsa-checking'
        return command