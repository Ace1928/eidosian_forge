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
def volume_mount_rest(self):
    body = {'nas.path': self.parameters['junction_path']}
    dummy, error = self.volume_rest_patch(body)
    if error:
        self.module.fail_json(msg='Error mounting volume %s with path "%s": %s' % (self.parameters['name'], self.parameters['junction_path'], to_native(error)), exception=traceback.format_exc())