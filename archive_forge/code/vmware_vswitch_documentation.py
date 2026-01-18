from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native

        Update the traffic shaping policy according to the parameters
        Args:
            spec: The vSwitch spec
            results: The results dict

        Returns: True if changes have been made, else false
        