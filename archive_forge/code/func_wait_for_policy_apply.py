from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def wait_for_policy_apply(self):
    """
        As the API is quite buggy and unstable there are cases where the policy still indicates pending changes
        even after the apply-policy task has finished. Such state usually goes away after few seconds,
        this function waits for the policy to achieve such a state for
        maximum of 60 seconds.

        """
    time.sleep(3)
    for x in range(0, 30):
        self.have = self.read_current_from_device()
        if self.have.apply is False:
            break
        time.sleep(2)
    return True