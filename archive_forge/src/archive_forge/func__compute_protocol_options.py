from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _compute_protocol_options(protocol_dict):
    cmd = ''
    for value in protocol_options.values():
        for subkey, subvalue in iteritems(value):
            if subvalue:
                cmd += '{0} '.format(subkey.replace('_', '-'))
    return cmd