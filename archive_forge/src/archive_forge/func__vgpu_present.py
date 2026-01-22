from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _vgpu_present(self, vm_obj):
    """
        Add vGPU profile to virtual machine.
        Args:
            vm_obj: Managed object of virtual machine
        Returns: Operation results and vGPU facts
        """
    result = {}
    vgpu_prfl = self.params['vgpu']
    vgpu_profile_name = self._get_vgpu_profiles_name(vm_obj, vgpu_prfl)
    if vgpu_profile_name is None:
        self.module.fail_json(msg="vGPU Profile '%s' does not exist." % vgpu_prfl)
    changed, failed = self._add_vgpu_profile_to_vm(vm_obj, vgpu_profile_name, vgpu_prfl)
    vgpu_fact = self._gather_vgpu_profile_facts(vm_obj)
    result = {'changed': changed, 'failed': failed, 'vgpu': vgpu_fact}
    return result