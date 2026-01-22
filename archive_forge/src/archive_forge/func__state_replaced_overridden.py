from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def _state_replaced_overridden(self, want, have, state):
    """ The command generator when state is replaced/overridden

        :rtype: A list
        :returns: the commands necessary to remove the current configuration
                  of the provided objects
        """
    commands = []
    requests = []
    if want and (not have):
        commands = [update_states(want, state)]
        requests = self.get_create_mclag_requests(want, want)
    elif not want and have:
        commands = [update_states(have, 'deleted')]
        requests = self.get_delete_all_mclag_domain_requests(have)
    elif want and have:
        add_command = {}
        del_command = {}
        delete_all = False
        if want['domain_id'] != have['domain_id']:
            del_command = have
            add_command = want
            delete_all = True
        else:
            have = have.copy()
            want = want.copy()
            delete_all_vlans = {'unique_ip': False, 'peer_gateway': False}
            delete_unspecified = True
            if state == 'replaced' and (not self.mclag_simple_attrs.intersection(remove_empties(want).keys())):
                delete_unspecified = False
            for option in ('unique_ip', 'peer_gateway'):
                have_cfg = {}
                want_cfg = {}
                if have.get(option) and have[option].get('vlans'):
                    have_cfg = have.pop(option)
                if want.get(option) and 'vlans' in want[option]:
                    want_cfg = want.pop(option)
                if want_cfg:
                    if have_cfg:
                        if not want_cfg['vlans']:
                            delete_all_vlans[option] = True
                            del_command[option] = have_cfg
                        else:
                            have_vlans = set(self.get_vlan_id_list(have_cfg['vlans']))
                            want_vlans = set(self.get_vlan_id_list(want_cfg['vlans']))
                            if have_vlans.intersection(want_vlans):
                                del_command[option] = {'vlans': self.get_vlan_range_list(list(have_vlans - want_vlans))}
                                if not del_command[option]['vlans']:
                                    del_command.pop(option)
                                add_command[option] = {'vlans': self.get_vlan_range_list(list(want_vlans - have_vlans))}
                                if not add_command[option]['vlans']:
                                    add_command.pop(option)
                            else:
                                delete_all_vlans[option] = True
                                del_command[option] = have_cfg
                                add_command[option] = want_cfg
                    elif want_cfg['vlans']:
                        add_command[option] = want_cfg
                elif have_cfg and delete_unspecified:
                    delete_all_vlans[option] = True
                    del_command[option] = have_cfg
            del_diff = get_diff(self.remove_default_entries(have), want, TEST_KEYS)
            for option in del_diff:
                if not want.get(option):
                    if delete_unspecified:
                        del_command[option] = del_diff[option]
                else:
                    if option == 'members' and want.get(option):
                        del_command[option] = del_diff[option]
                    if option == 'gateway_mac' and want.get(option):
                        del_command[option] = del_diff[option]
            diff = get_diff(want, have, TEST_KEYS)
            add_command.update(diff)
        if del_command:
            del_command['domain_id'] = have['domain_id']
            commands.extend(update_states(del_command, 'deleted'))
            if delete_all:
                requests = self.get_delete_all_mclag_domain_requests(del_command)
            else:
                if any(delete_all_vlans.values()):
                    del_command = deepcopy(del_command)
                for option in delete_all_vlans:
                    if delete_all_vlans[option]:
                        del_command[option]['vlans'] = None
                requests = self.get_delete_mclag_attribute_requests(del_command['domain_id'], del_command)
        if add_command:
            add_command['domain_id'] = want['domain_id']
            commands.extend(update_states(add_command, state))
            requests.extend(self.get_create_mclag_requests(add_command, add_command))
    return (commands, requests)