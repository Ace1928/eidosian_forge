from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _form_attr_cmd(self, key=None, attr=None, val=None, opr=True):
    """
        This function forms the command for leaf attribute.
        :param key: parent key.
        :param attr: attribute name
        :param value: value
        :param opr: True/False.
        :return: generated command.
        """
    return self._compute_command(key, attr=self._map_attrib(attr), val=val, opr=opr)