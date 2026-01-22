from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.ospfv3 import (
def _af_area_compare_lists(self, want, have):
    for attrib in ['filter_list', 'ranges']:
        wdict = want.get(attrib, {})
        hdict = have.get(attrib, {})
        for key, entry in iteritems(wdict):
            if entry != hdict.pop(key, {}):
                entry['area_id'] = want['area_id']
                self.addcmd(entry, 'area.{0}'.format(attrib), False)
        for entry in hdict.values():
            entry['area_id'] = have['area_id']
            self.addcmd(entry, 'area.{0}'.format(attrib), True)