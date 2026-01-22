from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxGroup:

    def __init__(self, group):
        self.group = dict()
        for k, v in group.items():
            if k == 'users' and isinstance(v, str):
                self.group['users'] = v.split(',')
            elif k == 'members':
                self.group['users'] = group['members']
            else:
                self.group[k] = v