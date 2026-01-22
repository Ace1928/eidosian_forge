from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_list_dict_param(self, attr, want, have, cmd=None, opr=True):
    """
        This function forms the set/delete commands based on the 'opr' type
        for attributes with in desired list of dictionary.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param cmd: commands to be prepend.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    h = []
    name = {'redistribute': 'route_type', 'neighbor': 'neighbor_id', 'range': 'address', 'md5': 'key_id', 'vlink': 'address'}
    leaf_dict = {'md5': 'md5_key', 'redistribute': ('metric', 'route_map', 'route_type', 'metric_type'), 'neighbor': ('priority', 'poll_interval', 'neighbor_id'), 'range': ('cost', 'address', 'substitute', 'not_advertise'), 'vlink': ('address', 'dead_interval', 'transmit_delay', 'hello_interval', 'retransmit_interval')}
    leaf = leaf_dict[attr]
    w = want.get(attr) or []
    if have:
        h = have.get(attr) or []
    if not opr and (not h):
        commands.append(self._compute_command(attr=attr, opr=opr))
    elif w:
        for w_item in w:
            for key, val in iteritems(w_item):
                if not cmd:
                    cmd = self._compute_command(opr=opr)
                h_item = self.search_obj_in_have(h, w_item, name[attr])
                if opr and key in leaf and (not _is_w_same(w_item, h_item, key)):
                    if key in ('route_type', 'neighbor_id', 'address', 'key_id'):
                        commands.append(cmd + attr + ' ' + str(val))
                    elif key == 'cost':
                        commands.append(cmd + attr + ' ' + w_item[name[attr]] + ' ' + key + ' ' + str(val))
                    elif key == 'not_advertise':
                        commands.append(cmd + attr + ' ' + w_item[name[attr]] + ' ' + key.replace('_', '-'))
                    elif key == 'md5_key':
                        commands.append(cmd + attr + ' ' + 'key-id' + ' ' + str(w_item[name[attr]]) + ' ' + key.replace('_', '-') + ' ' + w_item[key])
                    else:
                        commands.append(cmd + attr + ' ' + w_item[name[attr]] + ' ' + key.replace('_', '-') + ' ' + str(val))
                elif not opr and key in leaf and (not _in_target(h_item, key)):
                    if key in ('route_type', 'neighbor_id', 'address', 'key_id'):
                        commands.append(cmd + attr + ' ' + str(val))
                    else:
                        commands.append(cmd + attr + ' ' + w_item[name[attr]] + ' ' + key)
    return commands