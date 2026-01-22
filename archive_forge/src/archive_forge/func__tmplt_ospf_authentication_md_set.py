from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_authentication_md_set(config_data):
    command = _compute_command(config_data)
    auth = config_data['address_family']['authentication']
    if auth.get('message_digest') and auth['message_digest'].get('keychain'):
        command += ' authentication message-digest'
    elif auth.get('null_auth'):
        command += ' authentication null'
    return command