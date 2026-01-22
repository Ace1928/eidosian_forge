from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.ospfv3 import (
def _af_areas_compare(self, want, have):
    wareas = want.get('areas', {})
    hareas = have.get('areas', {})
    for name, entry in iteritems(wareas):
        self._af_area_compare(want=entry, have=hareas.pop(name, {}))
    for name, entry in iteritems(hareas):
        self._af_area_compare(want={}, have=entry)