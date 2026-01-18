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
def volume_rest_patch(self, body, query=None, uuid=None):
    if not uuid:
        uuid = self.parameters['uuid']
    if not uuid:
        self.module.fail_json(msg='Could not read UUID for volume %s in patch.' % self.parameters['name'])
    return rest_generic.patch_async(self.rest_api, 'storage/volumes', uuid, body, query=query, job_timeout=self.parameters['time_out'])