from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_child_param(self, w, h, key, opr=True):
    """
        This function invoke the function to extend commands
        based on the key.
        :param w: the desired configuration.
        :param h: the current configuration.
        :param key: attribute name.
        :param opr: operation.
        :return: list of commands.
        """
    commands = []
    if key in ('neighbor', 'redistribute'):
        commands.extend(self._render_list_dict_param(key, w, h, opr=opr))
    elif key in ('default_information', 'max_metric'):
        commands.extend(self._render_nested_dict_param(key, w, h, opr=opr))
    elif key in ('mpls_te', 'auto_cost', 'parameters', 'auto_cost'):
        commands.extend(self._render_dict_param(key, w, h, opr=opr))
    elif key in ('route_map', 'passive_interface', 'passive_interface_exclude'):
        commands.extend(self._render_list_param(key, w, h, opr=opr))
    elif key == 'areas':
        commands.extend(self._render_areas(key, w, h, opr=opr))
    elif key == 'timers':
        commands.extend(self._render_timers(key, w, h, opr=opr))
    elif key == 'distance':
        commands.extend(self._render_distance(key, w, h, opr=opr))
    return commands