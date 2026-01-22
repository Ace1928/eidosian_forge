from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def _compare_object_diff(self, want, have, object, object_type, parsers, val):
    temp_have = copy.copy(have)
    temp_want = copy.copy(want)
    if temp_have and temp_have.get(object) and temp_have[object].get(object_type):
        want_diff = self.get_list_diff(temp_want, temp_have, object, object_type)
        have_diff = [each for each in temp_have[object][object_type] if each not in temp_want[object][object_type]]
        if have_diff:
            temp_have[object].pop(object_type)
    else:
        have_diff = []
        want_diff = temp_want[object].get(object_type)
    temp_want[object][object_type] = want_diff
    if have_diff or (temp_have.get(object) and self.state in ('overridden', 'replaced')):
        if have_diff:
            temp_have[object] = {object_type: have_diff}
            self.compare(parsers, {}, temp_have)
    self.addcmd(temp_want, val, False)