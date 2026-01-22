from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.logging_global import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def __update_dict(self, datadict, key, nval=True):
    """Utility method that updates last subkey of
        `datadict` as identified by `key` to `nval`.
        """
    keys = key.split('.')
    if keys[0] not in datadict:
        datadict[keys[0]] = {}
    if keys[1] not in datadict[keys[0]]:
        datadict[keys[0]][keys[1]] = {}
    datadict[keys[0]][keys[1]].update({keys[2]: nval})