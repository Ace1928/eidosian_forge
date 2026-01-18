from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def validate_cvo_params(self):
    if self.parameters['use_latest_version'] is True and self.parameters['ontap_version'] != 'latest':
        self.module.fail_json(msg='ontap_version parameter not required when having use_latest_version as true')
    if self.parameters['is_ha'] is True and self.parameters['license_type'] == 'ha-cot-premium-byol':
        if self.parameters.get('platform_serial_number_node1') is None or self.parameters.get('platform_serial_number_node2') is None:
            self.module.fail_json(msg='both platform_serial_number_node1 and platform_serial_number_node2 parameters are requiredwhen having ha type as true and license_type as ha-cot-premium-byol')
    if self.parameters['is_ha'] is True and self.parameters['license_type'] == 'capacity-paygo':
        self.parameters['license_type'] = 'ha-capacity-paygo'