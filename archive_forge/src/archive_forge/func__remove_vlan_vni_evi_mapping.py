from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import dict_to_set
def _remove_vlan_vni_evi_mapping(self, want_dict):
    commands = []
    have_copy = self.have_now.copy()
    vlan = want_dict['vlan_id']
    for vlan_dict in have_copy:
        if vlan_dict['vlan_id'] == vlan:
            if 'member' in vlan_dict:
                commands.extend([self.vlan_parent.format(vlan), self._get_member_cmds(vlan_dict.get('member', {}), prefix='no')])
                vlan_dict.pop('member')
        if vlan_dict['vlan_id'] != vlan:
            delete_member = False
            have_vni = vlan_dict.get('member', {}).get('vni')
            have_evi = vlan_dict.get('member', {}).get('evi')
            if have_vni and have_vni == want_dict['member'].get('vni'):
                delete_member = True
            if have_evi and have_evi == want_dict['member'].get('evi'):
                delete_member = True
            if delete_member:
                commands.extend([self.vlan_parent.format(vlan_dict['vlan_id']), self._get_member_cmds(vlan_dict.get('member', {}), prefix='no')])
                self.have_now.remove(vlan_dict)
    return commands