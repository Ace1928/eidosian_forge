from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
def wait_for_instance(module, id):
    instance = None
    completed = False
    wait_timeout = time.time() + module.params.get('wait_time')
    while not completed and wait_timeout > time.time():
        try:
            completed = vsManager.wait_for_ready(id, 10, 2)
            if completed:
                instance = vsManager.get_instance(id)
        except Exception:
            completed = False
    return (completed, instance)