from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def SetVMHomeStorageProfile(self, vm, profile):
    """
        Set VM Home storage policy profile.

        :param vm: A virtual machine object.
        :type vm: VirtualMachine
        :param profile: A VMware Storage Policy profile.
        :type profile: pbm.profile.Profile
        :returns: VMware task object.
        :rtype: Task
        """
    spec = vim.vm.ConfigSpec()
    profileSpec = vim.vm.DefinedProfileSpec()
    profileSpec.profileId = profile.profileId.uniqueId
    spec.vmProfile = [profileSpec]
    return vm.ReconfigVM_Task(spec)