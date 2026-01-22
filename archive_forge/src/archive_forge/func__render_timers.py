from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
def _render_timers(self, item, config):
    """generate bgp timer related configuration"""
    keepalive = item['timers']['keepalive']
    holdtime = item['timers']['holdtime']
    neighbor = item['neighbor']
    if keepalive and holdtime:
        cmd = 'neighbor %s timers %s %s' % (neighbor, keepalive, holdtime)
        if not config or cmd not in config:
            return cmd
    else:
        raise ValueError('required both options for timers: keepalive and holdtime')