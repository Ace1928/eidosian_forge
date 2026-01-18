from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_and_configure_zapi(self):
    if self.parameters.get('storage_efficiency_mode'):
        self.module.fail_json(msg='Error: cannot set storage_efficiency_mode in ZAPI')
    if not self.parameters.get('start_ve_qos_policy'):
        self.parameters['start_ve_qos_policy'] = 'best-effort'
    if self.parameters.get('volume_name'):
        self.parameters['path'] = '/vol/' + self.parameters['volume_name']
        self.module.warn("ZAPI requires '/vol/' present in the volume path, updated path: %s" % self.parameters['path'])