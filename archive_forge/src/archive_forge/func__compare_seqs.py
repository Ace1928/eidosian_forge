from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.prefix_lists import (
def _compare_seqs(self, want, have):
    for wseq, wentry in iteritems(want):
        hentry = have.pop(wseq, {})
        if hentry != wentry:
            if hentry:
                if self.state == 'merged':
                    self._module.fail_json(msg='Cannot update existing sequence {0} of prefix list {1} with state merged. Please use state replaced or overridden.'.format(hentry['sequence'], hentry['name']))
                else:
                    self.addcmd(hentry, 'entry', negate=True)
            self.addcmd(wentry, 'entry')
    for hseq in have.values():
        self.addcmd(hseq, 'entry', negate=True)