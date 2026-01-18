from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import unquote
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_datacenter_by_name, find_cluster_by_name, \
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient

        Gather information about cluster
        