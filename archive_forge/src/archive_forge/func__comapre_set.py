from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.route_maps import (
def _comapre_set(self, want, have):
    parsers = ['set.as_path.prepend', 'set.as_path.match', 'set.bgp', 'set.community.graceful_shutdown', 'set.community.none', 'set.community.number', 'set.community.list', 'set.community.internet', 'set.distance', 'set.evpn', 'set.extcommunity.lbw', 'set.extcommunity.none', 'set.extcommunity.rt', 'set.extcommunity.soo', 'set.ip', 'set.ipv6', 'set.isis', 'set.local_pref', 'set.metric.value', 'set.metric_type', 'set.nexthop', 'set.origin', 'set.segment_index', 'set.tag', 'set.weight']
    w_set = want.pop('set', {})
    h_set = have.pop('set', {})
    for k, v in iteritems(w_set):
        self.compare(parsers=parsers, want={'entries': {'set': {k: v}}}, have={'entries': {'set': {k: h_set.pop(k, {})}}})
    for k, v in iteritems(h_set):
        self.compare(parsers=parsers, want={}, have={'entries': {'set': {k: v}}})