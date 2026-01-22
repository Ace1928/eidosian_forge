from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def _convert_date_to_epoch(module):
    try:
        unix_date = datetime.strptime(module.params['keep_current_until'], '%Y-%m-%d')
    except ValueError:
        module.fail_json(msg='Incorrect data format, should be YYYY-MM-DD')
    if unix_date < datetime.utcnow():
        module.warn('This value of `keep_current_until` will permanently delete objects as they are created. Using this date is not recommended')
    epoch_milliseconds = int((unix_date - datetime(1970, 1, 1)).total_seconds() * 1000)
    return epoch_milliseconds