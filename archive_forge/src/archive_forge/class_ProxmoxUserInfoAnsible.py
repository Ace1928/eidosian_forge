from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxUserInfoAnsible(ProxmoxAnsible):

    def get_user(self, userid):
        try:
            user = self.proxmox_api.access.users.get(userid)
        except Exception:
            self.module.fail_json(msg="User '%s' does not exist" % userid)
        user['userid'] = userid
        return ProxmoxUser(user)

    def get_users(self, domain=None):
        users = self.proxmox_api.access.users.get(full=1)
        users = [ProxmoxUser(user) for user in users]
        if domain:
            return [user for user in users if user.user['domain'] == domain]
        return users