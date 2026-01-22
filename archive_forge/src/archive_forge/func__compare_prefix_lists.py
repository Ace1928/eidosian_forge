from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.prefix_lists import (
def _compare_prefix_lists(self, afi, pk, w_list, have):
    parser = ['prefixlist.entry', 'prefixlist.resequence']
    for ek, ev in iteritems(w_list):
        if ek == 'name':
            continue
        h_child = {}
        if have.get('prefix_lists'):
            if have['prefix_lists'].get(pk):
                if have['prefix_lists'][pk].get(ek):
                    self._compare_seq(afi, w_list['entries'], have['prefix_lists'][pk][ek])
                for seq, seq_val in iteritems(have['prefix_lists'][pk][ek]):
                    h_child = {'afi': afi, 'prefix_lists': {'entries': {seq: seq_val}}}
                    self.compare(parsers=parser, want={}, have=h_child)
                have['prefix_lists'].pop(pk)
            else:
                self._compare_seq(afi, w_list['entries'], {})
        else:
            self._compare_seq(afi, w_list['entries'], {})