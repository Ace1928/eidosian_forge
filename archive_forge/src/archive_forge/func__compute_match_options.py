from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _compute_match_options(want_ace):
    cmd = ''
    if 'precedence' in want_ace:
        cmd += 'precedence {0} '.format(want_ace['precedence'])
    for x in ['dscp', 'packet_length', 'ttl']:
        if x in want_ace:
            opt_range = want_ace[x].get('range')
            if opt_range:
                cmd += '{0} range {1} {2} '.format(x.replace('_', '-'), opt_range['start'], opt_range['end'])
            else:
                for key, value in iteritems(want_ace[x]):
                    cmd += '{0} {1} {2} '.format(x.replace('_', '-'), key, value)
    for x in ('authen', 'capture', 'fragments', 'routing', 'log', 'log_input', 'icmp_off', 'destopts', 'hop_by_hop'):
        if x in want_ace:
            cmd += '{0} '.format(x.replace('_', '-'))
    return cmd