from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_vlink_auth(self, attr, key, want, have, address, cmd=None, opr=True):
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
    w = want.get(key) or {}
    if have:
        h = have.get(key) or {}
    cmd += attr.replace('_', '-') + ' ' + address + ' ' + key + ' '
    commands.extend(self._render_list_dict_param('md5', w, h, cmd, opr))
    return commands