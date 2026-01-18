from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def update_bgp_nbr_pg_prefix_limit_dict(pfx_lmt_conf):
    prefix_limit = {}
    if 'max-prefixes' in pfx_lmt_conf and pfx_lmt_conf['max-prefixes']:
        prefix_limit.update({'max_prefixes': pfx_lmt_conf['max-prefixes']})
    if 'prevent-teardown' in pfx_lmt_conf and pfx_lmt_conf['prevent-teardown']:
        prefix_limit.update({'prevent_teardown': pfx_lmt_conf['prevent-teardown']})
    if 'warning-threshold-pct' in pfx_lmt_conf and pfx_lmt_conf['warning-threshold-pct']:
        prefix_limit.update({'warning_threshold': pfx_lmt_conf['warning-threshold-pct']})
    if 'restart-timer' in pfx_lmt_conf and pfx_lmt_conf['restart-timer']:
        prefix_limit.update({'restart_timer': pfx_lmt_conf['restart-timer']})
    return prefix_limit