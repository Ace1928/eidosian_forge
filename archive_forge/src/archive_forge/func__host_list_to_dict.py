from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.snmp_server import (
def _host_list_to_dict(self, data):
    host_dict = {}
    host_data = deepcopy(data)
    for el in host_data['hosts']:
        tr = ''
        inf = ''
        if el.get('traps'):
            tr = 'traps'
        if el.get('informs'):
            inf = 'informs'
        host_dict.update({(el.get('host'), el.get('community'), el.get('version'), inf, tr, el.get('udp_port')): el})
    return host_dict