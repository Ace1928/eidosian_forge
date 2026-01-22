from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _link_local_query(v, value):
    v_ip = netaddr.IPAddress(str(v.ip))
    if v.version == 4:
        if ipaddr(str(v_ip), '169.254.0.0/16'):
            return value
    elif v.version == 6:
        if ipaddr(str(v_ip), 'fe80::/10'):
            return value