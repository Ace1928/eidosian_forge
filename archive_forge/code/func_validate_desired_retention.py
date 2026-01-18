from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from datetime import datetime, timedelta
import time
import copy
def validate_desired_retention(self, desired_retention, retention_unit):
    """Validates the specified desired retention.
            :param desired_retention: Desired retention of the snapshot
            :param retention_unit: Retention unit for snapshot
        """
    if retention_unit == 'hours' and (desired_retention < 1 or desired_retention > 744):
        self.module.fail_json(msg='Please provide a valid integer as the desired retention between 1 and 744.')
    elif retention_unit == 'days' and (desired_retention < 1 or desired_retention > 31):
        self.module.fail_json(msg='Please provide a valid integer as the desired retention between 1 and 31.')