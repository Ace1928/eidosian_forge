from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def _user_object_compare(self, want, have):
    user_obj = 'user_object'
    parsers = ['user_object.user', 'user_object.user_gp']
    add_obj_cmd = False
    for name, entry in iteritems(want):
        h_item = have.pop(name, {})
        if entry != h_item and name != 'object_type':
            if h_item and entry.get('group_object'):
                self.addcmd(entry, 'og_name', False)
                self._add_group_object_cmd(entry, h_item)
                continue
            if h_item:
                self._add_object_cmd(entry, h_item, user_obj, ['user', 'user_group'])
            else:
                add_obj_cmd = True
                self.addcmd(entry, 'og_name', False)
                self.compare(['description'], entry, h_item)
            if entry.get('group_object'):
                self._add_group_object_cmd(entry, h_item)
                continue
            if entry[user_obj].get('user'):
                self._compare_object_diff(entry, h_item, user_obj, 'user', ['user_object.user'], 'user_object.user')
            elif h_item and h_item[user_obj].get('user'):
                h_item[user_obj] = {'user': h_item[user_obj].get('user')}
                if not add_obj_cmd:
                    self.addcmd(entry, 'og_name', False)
                self.compare(parsers, {}, h_item)
            if entry[user_obj].get('user_group'):
                self._compare_object_diff(entry, h_item, user_obj, 'user_group', ['user_object.user_group'], 'user_object.user_gp')
            elif h_item and h_item[user_obj].get('user_group'):
                h_item[user_obj] = {'user_group': h_item[user_obj].get('user_group')}
                if not add_obj_cmd:
                    self.addcmd(entry, 'og_name', False)
                self.compare(parsers, {}, h_item)
    self.check_for_have_and_overidden(have)