from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, \
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient

        Get one/all virtual machines and related configurations information.
        