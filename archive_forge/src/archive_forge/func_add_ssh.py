from __future__ import absolute_import, division, print_function
import base64
import hashlib
import re
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def add_ssh(command, want, x=None):
    command.append('ip ssh pubkey-chain')
    if x:
        command.append('username %s' % want['name'])
        for item in x:
            command.append('key-hash %s' % item)
        command.append('exit')
    else:
        command.append('no username %s' % want['name'])
    command.append('exit')