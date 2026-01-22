from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ntp_global import (
def _compare_access_groups(self, want, have):
    w = want.get('access_group', {})
    h = have.get('access_group', {})
    for _parser in self.complex_parser[0:4]:
        i_want = w.get(_parser, {})
        i_have = h.get(_parser, {})
        for key, wanting in iteritems(i_want):
            haveing = i_have.pop(key, {})
            if wanting != haveing:
                self.addcmd(wanting, _parser)
        for key, haveing in iteritems(i_have):
            self.addcmd(haveing, _parser, negate=True)