from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _get_vgpu_profiles_name(self, vm_obj, vgpu_prfl):
    """
        Get matched vGPU profile object of ESXi host
        Args:
            vm_obj: Managed object of virtual machine
            vgpu_prfl: vGPU profile name
        Returns: vGPU profile object
        """
    vm_host = vm_obj.runtime.host
    vgpu_profiles = vm_host.config.sharedGpuCapabilities
    for vgpu_profile_name in vgpu_profiles:
        if vgpu_profile_name.vgpu == vgpu_prfl:
            return vgpu_profile_name
    return None