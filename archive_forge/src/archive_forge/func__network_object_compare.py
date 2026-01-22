from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def _network_object_compare(self, want, have):
    network_obj = 'network_object'
    parsers = ['network_object.host', 'network_object.address', 'network_object.ipv6_address', 'network_object.object']
    add_obj_cmd = False
    for name, entry in iteritems(want):
        h_item = have.pop(name, {})
        if entry != h_item and name != 'object_type':
            if h_item and entry.get('group_object'):
                self.addcmd(entry, 'og_name', False)
                self._add_group_object_cmd(entry, h_item)
                continue
            if h_item:
                self._add_object_cmd(entry, h_item, network_obj, ['address', 'host', 'ipv6_address', 'object'])
            else:
                add_obj_cmd = True
                self.addcmd(entry, 'og_name', False)
                self.compare(['description'], entry, h_item)
            if entry.get('group_object'):
                self._add_group_object_cmd(entry, h_item)
                continue
            if entry[network_obj].get('address'):
                self._compare_object_diff(entry, h_item, network_obj, 'address', parsers, 'network_object.address')
            elif h_item and h_item.get(network_obj) and h_item[network_obj].get('address'):
                h_item[network_obj] = {'address': h_item[network_obj].get('address')}
                if not add_obj_cmd:
                    self.addcmd(entry, 'og_name', False)
                self.compare(parsers, {}, h_item)
            if entry[network_obj].get('host'):
                self._compare_object_diff(entry, h_item, network_obj, 'host', parsers, 'network_object.host')
            elif h_item and h_item[network_obj].get('host'):
                h_item[network_obj] = {'host': h_item[network_obj].get('host')}
                if not add_obj_cmd:
                    self.addcmd(entry, 'og_name', False)
                self.compare(parsers, {}, h_item)
            if entry[network_obj].get('ipv6_address'):
                self._compare_object_diff(entry, h_item, network_obj, 'ipv6_address', parsers, 'network_object.ipv6_address')
            elif h_item and h_item.get(network_obj) and h_item[network_obj].get('ipv6_address'):
                h_item[network_obj] = {'ipv6_address': h_item[network_obj].get('ipv6_address')}
                if not add_obj_cmd:
                    self.addcmd(entry, 'og_name', False)
                self.compare(parsers, {}, h_item)
            if entry[network_obj].get('object'):
                self._compare_object_diff(entry, h_item, network_obj, 'object', parsers, 'network_object.object')
            elif h_item and h_item.get(network_obj) and h_item[network_obj].get('object'):
                h_item[network_obj] = {'object': h_item[network_obj].get('object')}
                if not add_obj_cmd:
                    self.addcmd(entry, 'og_name', False)
                self.compare(parsers, {}, h_item)
    self.check_for_have_and_overidden(have)