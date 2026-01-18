from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, wait_for_task, get_all_objs
from ansible.module_utils._text import to_native

        Get managed object of folder by name
        Returns: Managed object of folder by name

        