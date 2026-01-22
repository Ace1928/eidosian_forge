from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_aggregate_address(aggaddr):
    cmd = 'aggregate-address {value}'
    if aggaddr.get('as_set'):
        cmd += ' as-set'
    if aggaddr.get('as_confed_set'):
        cmd += ' as-confed-set'
    if aggaddr.get('summary_only'):
        cmd += ' summary-only'
    if aggaddr.get('route_policy'):
        cmd += ' route-policy {route_policy}'
    return cmd.format(**aggaddr)