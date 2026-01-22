from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _manage_vm_group(self):
    if self._check_if_vms_hosts_changed():
        need_reconfigure = False
        group = vim.cluster.VmGroup()
        group.name = self._group_name
        group.vm = self._group_obj.vm or []
        for vm in self._vm_obj_list:
            if self._operation == 'edit' and vm not in group.vm:
                group.vm.append(vm)
                need_reconfigure = True
            if self._operation == 'remove' and vm in group.vm:
                group.vm.remove(vm)
                need_reconfigure = True
        group_spec = vim.cluster.GroupSpec(info=group, operation='edit')
        config_spec = vim.cluster.ConfigSpecEx(groupSpec=[group_spec])
        if not self.module.check_mode and need_reconfigure:
            task = self._cluster_obj.ReconfigureEx(config_spec, modify=True)
            self.changed, dummy = wait_for_task(task)
        self._set_result(group)
        if self.changed:
            self.message = 'Updated vm group %s successfully' % self._group_name
        else:
            self.message = 'No update to vm group %s' % self._group_name
    else:
        self.changed = False
        self.message = 'No update to vm group %s' % self._group_name