from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import time
def validate_rest(self, modify):
    if modify.get('is_enabled') is False:
        error = 'Error: disable service processor network status not allowed in REST'
        self.module.fail_json(msg=error)
    if modify.get('is_enabled') and len(modify) == 1:
        error = 'Error: enable service processor network requires dhcp or ip_address,netmask,gateway details in REST.'
        self.module.fail_json(msg=error)