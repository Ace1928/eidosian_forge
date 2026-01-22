from __future__ import absolute_import, division, print_function
import platform
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
class OFacts(FactsBase):

    def populate(self):
        device = get_device(self.module)
        facts = dict(device.facts)
        if '2RE' in facts:
            facts['has_2RE'] = facts['2RE']
            del facts['2RE']
        facts['version_info'] = dict(facts['version_info'])
        if 'junos_info' in facts:
            for key, value in facts['junos_info'].items():
                if 'object' in value:
                    value['object'] = dict(value['object'])
        return facts