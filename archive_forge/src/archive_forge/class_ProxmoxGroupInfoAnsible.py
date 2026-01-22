from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxGroupInfoAnsible(ProxmoxAnsible):

    def get_group(self, groupid):
        try:
            group = self.proxmox_api.access.groups.get(groupid)
        except Exception:
            self.module.fail_json(msg="Group '%s' does not exist" % groupid)
        group['groupid'] = groupid
        return ProxmoxGroup(group)

    def get_groups(self):
        groups = self.proxmox_api.access.groups.get()
        return [ProxmoxGroup(group) for group in groups]