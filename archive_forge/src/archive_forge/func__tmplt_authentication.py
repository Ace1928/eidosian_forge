from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_authentication(data):
    auth = data.get('authentication')
    cmd = 'ip ospf authentication'
    if auth.get('enable') is False:
        cmd = 'no ' + cmd
    elif auth.get('message_digest'):
        cmd += ' message-digest'
    elif auth.get('null_auth'):
        cmd += ' null'
    return cmd