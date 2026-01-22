from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.ospfv2 import (
def _ospf_list_to_dict(self, entry):
    for _pid, proc in iteritems(entry):
        for area in proc.get('areas', []):
            area['ranges'] = {entry['prefix']: entry for entry in area.get('ranges', [])}
            area['filter_list'] = {entry['direction']: entry for entry in area.get('filter_list', [])}
        mpls_areas = {entry['area_id']: entry for entry in proc.get('mpls', {}).get('traffic_eng', {}).get('areas', [])}
        if mpls_areas:
            proc['mpls']['traffic_eng']['areas'] = mpls_areas
        proc['areas'] = {entry['area_id']: entry for entry in proc.get('areas', [])}
        proc['summary_address'] = {entry['prefix']: entry for entry in proc.get('summary_address', [])}
        proc['redistribute'] = {(entry.get('id'), entry['protocol']): entry for entry in proc.get('redistribute', [])}
        if 'vrfs' in proc:
            proc['vrfs'] = {entry['vrf']: entry for entry in proc.get('vrfs', [])}
            self._ospf_list_to_dict(proc['vrfs'])