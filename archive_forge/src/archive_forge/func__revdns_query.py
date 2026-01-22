from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _revdns_query(v):
    v_ip = netaddr.IPAddress(str(v.ip))
    return v_ip.reverse_dns