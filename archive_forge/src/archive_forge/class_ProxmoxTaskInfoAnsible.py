from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxTaskInfoAnsible(ProxmoxAnsible):

    def get_task(self, upid, node):
        tasks = self.get_tasks(node)
        for task in tasks:
            if task.info['upid'] == upid:
                return [task]

    def get_tasks(self, node):
        tasks = self.proxmox_api.nodes(node).tasks.get()
        return [ProxmoxTask(task) for task in tasks]