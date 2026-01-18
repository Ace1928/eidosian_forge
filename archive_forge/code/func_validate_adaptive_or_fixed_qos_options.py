from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_adaptive_or_fixed_qos_options(self):
    error = None
    if 'fixed_qos_options' in self.parameters:
        fixed_options = ['max_throughput_iops', 'max_throughput_mbps', 'min_throughput_iops', 'min_throughput_mbps']
        if not any((x in self.na_helper.filter_out_none_entries(self.parameters['fixed_qos_options']) for x in fixed_options)):
            error = True
    elif self.parameters.get('fixed_qos_options', self.parameters.get('adaptive_qos_options')) is None:
        error = True
    return error