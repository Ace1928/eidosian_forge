from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _bool_to_str(self, val):
    """
        This function converts the bool value into string.
        :param val: bool value.
        :return: enable/disable.
        """
    return 'enable' if str(val) == 'True' else 'disable' if str(val) == 'False' else val