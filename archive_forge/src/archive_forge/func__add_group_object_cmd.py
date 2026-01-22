from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def _add_group_object_cmd(self, want, have):
    if have and have.get('group_object'):
        want['group_object'] = list(set(want.get('group_object')) - set(have.get('group_object')))
        have['group_object'] = list(set(have.get('group_object')) - set(want.get('group_object')))
    for each in want['group_object']:
        self.compare(['group_object'], {'group_object': each}, dict())
    if (self.state == 'replaced' or self.state == 'overridden') and have and have.get('group_object'):
        for each in have['group_object']:
            self.compare(['group_object'], dict(), {'group_object': each})