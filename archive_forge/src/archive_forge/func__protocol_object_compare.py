from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def _protocol_object_compare(self, want, have):
    protocol_obj = 'protocol_object'
    for name, entry in iteritems(want):
        h_item = have.pop(name, {})
        if entry != h_item and name != 'object_type':
            if h_item and entry.get('group_object'):
                self.addcmd(entry, 'og_name', False)
                self._add_group_object_cmd(entry, h_item)
                continue
            if h_item:
                self._add_object_cmd(entry, h_item, protocol_obj, ['protocol'])
            else:
                self.addcmd(entry, 'og_name', False)
                self.compare(['description'], entry, h_item)
            if entry.get('group_object'):
                self._add_group_object_cmd(entry, h_item)
                continue
            if entry[protocol_obj].get('protocol'):
                self._compare_object_diff(entry, h_item, protocol_obj, 'protocol', [protocol_obj], protocol_obj)
    self.check_for_have_and_overidden(have)