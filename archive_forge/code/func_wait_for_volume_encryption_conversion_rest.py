from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def wait_for_volume_encryption_conversion_rest(self):
    api = 'storage/volumes'
    query = {'name': self.parameters['name'], 'fields': 'encryption'}
    error = self.wait_for_task_completion_rest(api, query, self.check_volume_encryption_conversion_state)
    if error:
        self.module.fail_json(msg='Error getting volume encryption_conversion status: %s' % to_native(error), exception=traceback.format_exc())