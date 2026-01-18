from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible_collections.community.vmware.plugins.module_utils.vmware import (

        Gather facts about VM's vGPU profile settings
        Args:
            vm_obj: Managed object of virtual machine
        Returns: list of vGPU profiles with facts
        