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
def volume_unmount_rest(self, fail_on_error=True):
    body = {'nas.path': ''}
    dummy, error = self.volume_rest_patch(body)
    if error and fail_on_error:
        self.module.fail_json(msg='Error unmounting volume %s with path "%s": %s' % (self.parameters['name'], self.parameters.get('junction_path'), to_native(error)), exception=traceback.format_exc())
    return error