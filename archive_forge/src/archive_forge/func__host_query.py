from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _host_query(v):
    if v.size == 1:
        return str(v)
    elif v.size > 1:
        if v.ip != v.network or not v.broadcast:
            return str(v.ip) + '/' + str(v.prefixlen)