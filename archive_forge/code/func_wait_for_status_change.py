from __future__ import absolute_import, division, print_function
import time
import re
from collections import namedtuple
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import python_2_unicode_compatible
def wait_for_status_change(self, current_status):
    running_status = self.get_status()
    if running_status.value != current_status.value or current_status.value == StatusValue.EXECUTION_FAILED:
        return running_status
    loop_count = 0
    while running_status.value == current_status.value:
        if loop_count >= self._status_change_retry_count:
            self.exit_fail('waited too long for monit to change state', running_status)
        loop_count += 1
        time.sleep(0.5)
        validate = loop_count % 2 == 0
        running_status = self.get_status(validate)
    return running_status