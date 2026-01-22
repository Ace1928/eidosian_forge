from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.bgp_global import (
def _has_af(self, vrf=None, neighbor=None):
    """Determine if the given vrf + neighbor
           combination has AF configurations.

        :params vrf: vrf name
        :params neighbor: neighbor name
        :returns: bool
        """
    has_af = False
    if self._af_data:
        vrf_af_data = self._af_data.get('vrf', {})
        global_af_data = self._af_data.get('global', set())
        if vrf:
            vrf_nbr_has_af = vrf_af_data.get(vrf, {}).get('nbrs', set())
            vrf_has_af = vrf_af_data.get(vrf, {}).get('has_af', False)
            if neighbor and neighbor in vrf_nbr_has_af:
                has_af = True
            elif vrf_nbr_has_af or vrf_has_af:
                has_af = True
        elif neighbor and neighbor in global_af_data:
            has_af = True
    return has_af