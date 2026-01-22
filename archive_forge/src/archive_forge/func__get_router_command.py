from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _get_router_command(inst):
    command = ''
    if inst.get('vrf') and inst.get('vrf') != 'default':
        command = 'router ospf ' + str(inst['process_id']) + ' vrf ' + inst['vrf']
    else:
        command = 'router ospf ' + str(inst['process_id'])
    return command