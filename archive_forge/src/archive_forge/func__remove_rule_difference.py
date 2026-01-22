from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _remove_rule_difference(self, rules):
    if rules is None or self.have.rules is None:
        return
    have_rules = set(self.have.rules)
    want_rules = set(rules)
    removable = have_rules.difference(want_rules)
    for remove in removable:
        self.remove_rule_from_device(remove)