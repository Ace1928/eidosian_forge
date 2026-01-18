from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native

        Function to return list of VMKernels
        Args:
            host_system: Host system managed object
            service_type: Name of service type

        Returns: List of VMKernel which belongs to that service type

        