from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.arista.eos.plugins.module_utils.network.eos.eos import (
def verify_state(updates, module):
    want, have = updates
    invalid_state = [('http', 'httpServer'), ('https', 'httpsServer'), ('local_http', 'localHttpServer'), ('socket', 'unixSocketServer')]
    timeout = module.params['timeout']
    state = module.params['state']
    while invalid_state:
        out = run_commands(module, ['show management api http-commands | json'])
        for index, item in enumerate(invalid_state):
            want_key, eapi_key = item
            if want[want_key] is not None:
                if want[want_key] == out[0][eapi_key]['running']:
                    del invalid_state[index]
            elif state == 'stopped':
                if not out[0][eapi_key]['running']:
                    del invalid_state[index]
            else:
                del invalid_state[index]
        time.sleep(1)
        timeout -= 1
        if timeout == 0:
            module.fail_json(msg='timeout expired before eapi running state changed')