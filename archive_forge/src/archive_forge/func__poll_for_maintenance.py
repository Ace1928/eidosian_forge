from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _poll_for_maintenance(self):
    for i in range(0, 300):
        time.sleep(2)
        host = self.get_host(refresh=True)
        if not host:
            return None
        elif host['resourcestate'] != 'PrepareForMaintenance':
            return host
    self.fail_json(msg='Polling for maintenance timed out')