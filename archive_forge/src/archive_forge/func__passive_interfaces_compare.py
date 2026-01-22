from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv2 import (
def _passive_interfaces_compare(self, want, have):
    parsers = ['passive_interfaces.default', 'passive_interfaces.interface']
    h_pi = None
    for k, v in iteritems(want['passive_interfaces']):
        h_pi = have.get('passive_interfaces', {})
        if h_pi.get(k) and h_pi.get(k) != v:
            for each in v['name']:
                h_interface_name = h_pi[k].get('name', [])
                if each not in h_interface_name:
                    temp = {'interface': {each: each}, 'set_interface': v['set_interface']}
                    self.compare(parsers=parsers, want={'passive_interfaces': temp}, have=dict())
                else:
                    h_interface_name.pop(each)
        elif not h_pi:
            if k == 'interface':
                for each in v['name']:
                    temp = {'interface': {each: each}, 'set_interface': v['set_interface']}
                    self.compare(parsers=parsers, want={'passive_interfaces': temp}, have=dict())
            elif k == 'default':
                self.compare(parsers=parsers, want={'passive_interfaces': {'default': True}}, have=dict())
        else:
            h_pi.pop(k)
    if (self.state == 'replaced' or self.state == 'overridden') and h_pi:
        if h_pi.get('default') or h_pi.get('interface'):
            for k, v in iteritems(h_pi):
                if k == 'interface':
                    for each in v['name']:
                        temp = {'interface': {each: each}, 'set_interface': not v['set_interface']}
                        self.compare(parsers=parsers, want={'passive_interfaces': temp}, have=dict())
                elif k == 'default':
                    self.compare(parsers=parsers, want=dict(), have={'passive_interfaces': {'default': True}})