from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.snmp_server import (
def _compare_hosts(self, want, have):
    wdict = get_from_dict(want, 'hosts') or {}
    hdict = get_from_dict(have, 'hosts') or {}
    for key, entry in iteritems(wdict):
        self.compare(parsers='hosts', want={'hosts': {key: entry}}, have={'hosts': {key: hdict.pop(key, {})}})
    for key, entry in iteritems(hdict):
        self.compare(parsers='hosts', want={}, have={'hosts': {key: entry}})