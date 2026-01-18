from __future__ import (absolute_import, division, print_function)
import json
import re
import sys
import datetime
import time
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
def wait_to_complete_state_transition(self, resource, stable_states, force_wait=False):
    wait = self.module.params['wait']
    if not (wait or force_wait):
        return
    wait_timeout = self.module.params['wait_timeout']
    wait_sleep_time = self.module.params['wait_sleep_time']
    time.sleep(wait_sleep_time)
    start = datetime.datetime.utcnow()
    end = start + datetime.timedelta(seconds=wait_timeout)
    while datetime.datetime.utcnow() < end:
        self.module.debug('We are going to wait for the resource to finish its transition')
        state = self.fetch_state(resource)
        if state in stable_states:
            self.module.debug('It seems that the resource is not in transition anymore.')
            self.module.debug('load-balancer in state: %s' % self.fetch_state(resource))
            break
        time.sleep(wait_sleep_time)
    else:
        self.module.fail_json(msg='Server takes too long to finish its transition')