from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def _add_object_cmd(self, want, have, object, object_elements):
    obj_cmd_added = False
    for each in object_elements:
        want_element = want[object].get(each) if want.get(object) else want
        have_element = have[object].get(each) if have.get(object) else have
        if want_element and isinstance(want_element, list) and isinstance(want_element[0], dict):
            if want_element and have_element and (want_element != have_element):
                if not obj_cmd_added:
                    self.addcmd(want, 'og_name', False)
                    self.compare(['description'], want, have)
                    obj_cmd_added = True
        elif want_element and have_element and (set(want_element) != set(have_element)):
            if not obj_cmd_added:
                self.addcmd(want, 'og_name', False)
                self.compare(['description'], want, have)
                obj_cmd_added = True