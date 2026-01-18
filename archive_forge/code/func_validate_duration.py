from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def validate_duration(self, name, duration):
    if duration:
        duration_units = ['ns', 'us', 'ms', 's', 'm', 'h']
        if not any((duration.endswith(suffix) for suffix in duration_units)):
            duration = '{0}s'.format(duration)
    return duration