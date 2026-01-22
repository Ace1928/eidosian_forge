from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _absent_changed_options(self):
    diff = Difference(self.want, self.have)
    absentables = Parameters.absentables
    changed = dict()
    for k in absentables:
        change = diff.compare(k)
        if change is None:
            continue
        elif isinstance(change, dict):
            changed.update(change)
        else:
            changed[k] = change
    if changed:
        self.changes = UsableChanges(params=changed)
        return True
    return False