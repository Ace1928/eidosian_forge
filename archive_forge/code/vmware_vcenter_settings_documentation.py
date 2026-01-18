from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, option_diff, vmware_argument_spec
from ansible.module_utils._text import to_native
Manage settings for a vCenter server