from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.ntp_global import (
def _servernames(self, haved):
    servernames = []
    for k, have in iteritems(haved):
        for sk, sval in iteritems(have):
            if sk == 'server' and sval not in ['0.pool.ntp.org', '1.pool.ntp.org', '2.pool.ntp.org']:
                if sval not in servernames:
                    servernames.append(sval)
    return servernames