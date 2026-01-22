from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def _convert_to_millisecs(hour):
    if hour[-2:] == 'AM' and hour[:2] == '12':
        return 0
    elif hour[-2:] == 'AM':
        return int(hour[:-2]) * 3600000
    elif hour[-2:] == 'PM' and hour[:2] == '12':
        return 43200000
    return (int(hour[:-2]) + 12) * 3600000