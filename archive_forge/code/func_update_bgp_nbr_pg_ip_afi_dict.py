from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def update_bgp_nbr_pg_ip_afi_dict(ip_afi_conf):
    ip_afi = {}
    if 'default-policy-name' in ip_afi_conf and ip_afi_conf['default-policy-name']:
        ip_afi.update({'default_policy_name': ip_afi_conf['default-policy-name']})
    if 'send-default-route' in ip_afi_conf and ip_afi_conf['send-default-route']:
        ip_afi.update({'send_default_route': ip_afi_conf['send-default-route']})
    return ip_afi