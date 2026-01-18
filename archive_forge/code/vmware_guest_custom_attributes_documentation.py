from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
Check the existing custom attributes and values.

        In the function, the below processing is executed.

        Gather the existing custom attributes from the virtual machine and make update_custom_attributes for updating
        if it has differences between the existing configuration and the user_fields.

        And set diff key for checking between before and after configuration to self.diff_config.

        Args:
            vm (vim.VirtualMachine): The managed object of a virtual machine.
            user_fields (list): list of the specified custom attributes by user.
        