from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote_plus
def waitForNoTask(client, name, timeout):
    currentTimeout = timeout
    while client.get('/ip/{0}/task'.format(quote_plus(name)), function='genericMoveFloatingIp', status='todo'):
        time.sleep(1)
        currentTimeout -= 1
        if currentTimeout < 0:
            return False
    return True