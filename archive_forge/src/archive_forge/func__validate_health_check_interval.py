from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def _validate_health_check_interval(module):
    error_message = None
    device_health = module.params.get('device_health')
    if device_health:
        hci = device_health.get('health_check_interval')
        hciu = device_health.get('health_check_interval_unit')
        if hci and (not hciu):
            error_message = HEALTH_CHECK_UNIT_REQUIRED
        if hciu and (not hci):
            error_message = HEALTH_CHECK_INTERVAL_REQUIRED
        if hciu and hci:
            if hciu == 'Hourly' and (hci < 1 or hci > 23):
                error_message = HEALTH_CHECK_INTERVAL_INVALID.format(hciu)
            if hciu == 'Minutes' and (hci < 1 or hci > 59):
                error_message = HEALTH_CHECK_INTERVAL_INVALID.format(hciu)
    return error_message