from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.ntp_global import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.utils.utils import (
def _authentication_keys_compare(self, want, have):
    w = want.pop('authentication_keys', {})
    h = have.pop('authentication_keys', {})
    for name, entry in iteritems(w):
        h_key = {}
        if h.get(name):
            h_key = {'authentication_keys': h.pop(name)}
        self.compare(parsers='authentication_keys', want={'authentication_keys': entry}, have=h_key)
    for name, entry in iteritems(h):
        self.compare(parsers='authentication_keys', want={}, have={'authentication_keys': entry})