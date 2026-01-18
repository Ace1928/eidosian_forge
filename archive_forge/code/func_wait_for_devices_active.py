from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def wait_for_devices_active(module, packet_conn, watched_devices):
    wait_timeout = module.params.get('wait_timeout')
    wait_timeout = time.time() + wait_timeout
    refreshed = watched_devices
    while wait_timeout > time.time():
        refreshed = refresh_device_list(module, packet_conn, watched_devices)
        if all((d.state == 'active' for d in refreshed)):
            return refreshed
        time.sleep(5)
    raise Exception('Waiting for state "active" timed out for devices: %s' % [d.hostname for d in refreshed if d.state != 'active'])