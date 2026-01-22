from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _address_prefix_query(v):
    if v.size > 2 and v.ip in (v.network, v.broadcast):
        return False
    return str(v.ip) + '/' + str(v.prefixlen)