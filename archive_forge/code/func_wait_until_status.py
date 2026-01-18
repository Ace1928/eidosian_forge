from __future__ import absolute_import, division, print_function
import csv
import socket
import time
from string import Template
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
def wait_until_status(self, pxname, svname, status):
    """
        Wait for a service to reach the specified status. Try RETRIES times
        with INTERVAL seconds of sleep in between. If the service has not reached
        the expected status in that time, the module will fail. If the service was
        not found, the module will fail.
        """
    for i in range(1, self.wait_retries):
        state = self.get_state_for(pxname, svname)
        if status in state[0]['status']:
            if not self._drain or state[0]['scur'] == '0':
                return True
        time.sleep(self.wait_interval)
    self.module.fail_json(msg="server %s/%s not status '%s' after %d retries. Aborting." % (pxname, svname, status, self.wait_retries))