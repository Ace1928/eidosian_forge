from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxUser:

    def __init__(self, user):
        self.user = dict()
        for k, v in user.items():
            if k == 'enable':
                self.user['enabled'] = proxmox_to_ansible_bool(user['enable'])
            elif k == 'userid':
                self.user['user'] = user['userid'].split('@')[0]
                self.user['domain'] = user['userid'].split('@')[1]
                self.user[k] = v
            elif k in ['groups', 'tokens'] and (v == '' or v is None):
                self.user[k] = []
            elif k == 'groups' and isinstance(v, str):
                self.user['groups'] = v.split(',')
            elif k == 'tokens' and isinstance(v, list):
                for token in v:
                    if 'privsep' in token:
                        token['privsep'] = proxmox_to_ansible_bool(token['privsep'])
                self.user['tokens'] = v
            elif k == 'tokens' and isinstance(v, dict):
                self.user['tokens'] = list()
                for tokenid, tokenvalues in v.items():
                    t = tokenvalues
                    t['tokenid'] = tokenid
                    if 'privsep' in tokenvalues:
                        t['privsep'] = proxmox_to_ansible_bool(tokenvalues['privsep'])
                    self.user['tokens'].append(t)
            else:
                self.user[k] = v