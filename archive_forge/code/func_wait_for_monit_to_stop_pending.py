from __future__ import absolute_import, division, print_function
import time
import re
from collections import namedtuple
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import python_2_unicode_compatible
def wait_for_monit_to_stop_pending(self, current_status=None):
    """Fails this run if there is no status or it's pending/initializing for timeout"""
    timeout_time = time.time() + self.timeout
    if not current_status:
        current_status = self.get_status()
    waiting_status = [StatusValue.MISSING, StatusValue.INITIALIZING, StatusValue.DOES_NOT_EXIST]
    while current_status.is_pending or current_status.value in waiting_status:
        if time.time() >= timeout_time:
            self.exit_fail('waited too long for "pending", or "initiating" status to go away', current_status)
        time.sleep(5)
        current_status = self.get_status(validate=True)
    return current_status