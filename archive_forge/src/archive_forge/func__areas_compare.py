from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv2 import (
def _areas_compare(self, want, have):
    wareas = want.get('areas', {})
    hareas = have.get('areas', {})
    for name, entry in iteritems(wareas):
        self._area_compare(want=entry, have=hareas.pop(name, {}))
    for name, entry in iteritems(hareas):
        self._area_compare(want={}, have=entry)