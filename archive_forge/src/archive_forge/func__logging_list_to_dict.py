from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.logging_global import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def _logging_list_to_dict(self, data):
    """Convert all list to dicts to dicts
        of dicts and substitute severity values
        """
    tmp = deepcopy(data)
    pkey = {'hosts': 'host', 'facilities': 'facility'}
    for k in ('hosts', 'facilities'):
        if k in tmp:
            for x in tmp[k]:
                if 'severity' in x:
                    x['severity'] = self._sev_map[x['severity']]
            tmp[k] = {i[pkey[k]]: i for i in tmp[k]}
    for k in ('console', 'history', 'logfile', 'module', 'monitor'):
        if 'severity' in tmp.get(k, {}):
            tmp[k]['severity'] = self._sev_map[tmp[k]['severity']]
    return tmp