from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver

        POST does not accept access_control and propagation_mode at the ACL level, these are global values for all ACLs.
        Since the user could have a list of ACLs with mixed property we should useP OST the create the SD and 1 group of ACLs
        then loop over the remaining ACLS.
        