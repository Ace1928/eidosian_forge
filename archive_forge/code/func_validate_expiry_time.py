from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def validate_expiry_time(self, expiry_time):
    """Validates the specified expiry_time"""
    try:
        datetime.strptime(expiry_time, '%m/%d/%Y %H:%M')
    except ValueError:
        error_msg = 'expiry_time: %s, not in MM/DD/YYYY HH:MM format.' % expiry_time
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)