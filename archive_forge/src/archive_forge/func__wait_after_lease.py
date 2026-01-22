from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _wait_after_lease(self):
    if self.param('lease') and self.param('wait_after_lease') != 0:
        time.sleep(self.param('wait_after_lease'))