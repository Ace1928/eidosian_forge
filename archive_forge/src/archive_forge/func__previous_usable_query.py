from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _previous_usable_query(v, vtype):
    if vtype == 'address':
        raise AnsibleFilterError('Not a network address')
    elif vtype == 'network':
        if v.size > 1:
            first_usable, last_usable = _first_last(v)
            previous_ip = int(netaddr.IPAddress(int(v.ip) - 1))
            if previous_ip >= first_usable and previous_ip <= last_usable:
                return str(netaddr.IPAddress(int(v.ip) - 1))