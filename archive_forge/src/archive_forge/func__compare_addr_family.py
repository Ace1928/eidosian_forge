from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.ospf_interfaces import (
def _compare_addr_family(self, want, have):
    wdict = want.get('address_family', {})
    hdict = have.get('address_family', {})
    wname = want.get('name')
    hname = have.get('name')
    for name, entry in iteritems(wdict):
        for key, param in iteritems(entry):
            w_addr = {'afi': name, key: param}
            h_addr = {}
            if hdict.get(name):
                h_addr = {'afi': name, key: hdict[name].pop(key, {})}
            w = {'name': wname, 'address_family': w_addr}
            h = {'name': hname, 'address_family': h_addr}
            self.compare(parsers=self.parsers, want=w, have=h)
    for name, entry in iteritems(hdict):
        for key, param in iteritems(entry):
            h_addr = {'afi': name, key: param}
            w_addr = {}
            w = {'name': wname, 'address_family': w_addr}
            h = {'name': hname, 'address_family': h_addr}
            self.compare(parsers=self.parsers, want=w, have=h)