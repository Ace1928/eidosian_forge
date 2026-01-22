from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.route_maps import (
def _compare_match(self, want, have):
    parsers = ['match.aggregate_role', 'match.as', 'match.as_path', 'match.community.instances', 'match.community.list', 'match.extcommunity', 'match.invert.aggregate_role', 'match.invert.as_path', 'match.invert.community.instances', 'match.invert.community.list', 'match.invert.extcommunity', 'match.interface', 'match.ip', 'match.ipaddress', 'match.ipv6', 'match.ipv6address', 'match.largecommunity', 'match.isis', 'match.local_pref', 'match.metric', 'match.metric_type', 'match.route_type', 'match.routerid', 'match.source_protocol', 'match.tag']
    w_match = want.pop('match', {})
    h_match = have.pop('match', {})
    for k, v in iteritems(w_match):
        if k in ['ip', 'ipv6']:
            for k_ip, v_ip in iteritems(v):
                if h_match.get(k):
                    h = {k_ip: h_match[k].pop(k_ip, {})}
                else:
                    h = {}
                self.compare(parsers=['match.ip', 'match.ipaddress', 'match.ipv6address', 'match.ipv6'], want={'entries': {'match': {k: {k_ip: v_ip}}}}, have={'entries': {'match': {k: h}}})
            h_match.pop(k, {})
            continue
        self.compare(parsers=parsers, want={'entries': {'match': {k: v}}}, have={'entries': {'match': {k: h_match.pop(k, {})}}})
    for k, v in iteritems(h_match):
        if k in ['ip', 'ipv6']:
            for hk, hv in iteritems(v):
                self.compare(parsers=['match.ip', 'match.ipaddress', 'match.ipv6address', 'match.ipv6'], want={}, have={'entries': {'match': {k: {hk: hv}}}})
            continue
        self.compare(parsers=parsers, want={}, have={'entries': {'match': {k: v}}})