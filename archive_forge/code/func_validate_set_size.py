from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.storage.hpe3par import hpe3par
def validate_set_size(raid_type, set_size):
    if raid_type:
        set_size_array = client.HPE3ParClient.RAID_MAP[raid_type]['set_sizes']
        if set_size in set_size_array:
            return True
    return False