from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv3 import (
def _area_compare_filters(self, wantd, haved):
    for name, entry in iteritems(wantd):
        h_item = haved.pop(name, {})
        if entry != h_item and name == 'filter_list':
            filter_list_entry = {}
            filter_list_entry['area_id'] = wantd['area_id']
            if h_item:
                li_diff = [item for item in entry + h_item if item not in entry or item not in h_item]
            else:
                li_diff = entry
            filter_list_entry['filter_list'] = li_diff
            self.addcmd(filter_list_entry, 'area.filter_list', False)
    for name, entry in iteritems(haved):
        if name == 'filter_list':
            self.addcmd(entry, 'area.filter_list', True)