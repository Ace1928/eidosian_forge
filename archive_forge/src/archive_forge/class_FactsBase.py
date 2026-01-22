from __future__ import absolute_import, division, print_function
import platform
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
class FactsBase(object):

    def __init__(self, module):
        self.module = module
        self.facts = dict()
        self.warnings = []

    def populate(self):
        raise NotImplementedError

    def cli(self, command):
        reply = command(self.module, command)
        output = reply.find('.//output')
        if not output:
            self.module.fail_json(msg='failed to retrieve facts for command %s' % command)
        return to_text(output.text).strip()

    def rpc(self, rpc):
        return exec_rpc(self.module, tostring(Element(rpc)))

    def get_text(self, ele, tag):
        try:
            return to_text(ele.find(tag).text).strip()
        except AttributeError:
            pass