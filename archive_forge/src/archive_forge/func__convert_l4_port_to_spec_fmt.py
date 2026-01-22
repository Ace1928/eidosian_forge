from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.l3_acls.l3_acls import L3_aclsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
@staticmethod
def _convert_l4_port_to_spec_fmt(l4_port):
    spec_fmt = {}
    if l4_port is not None:
        if isinstance(l4_port, str) and '..' in l4_port:
            l4_port = [int(i) for i in l4_port.split('..')]
            if l4_port[0] == L4_PORT_START:
                spec_fmt['lt'] = l4_port[1]
            elif l4_port[1] == L4_PORT_END:
                spec_fmt['gt'] = l4_port[0]
            else:
                spec_fmt['range'] = {'begin': l4_port[0], 'end': l4_port[1]}
        else:
            spec_fmt['eq'] = int(l4_port)
    return spec_fmt