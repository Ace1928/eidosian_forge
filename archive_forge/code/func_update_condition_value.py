from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def update_condition_value(self, name, condition):
    """requires an expected value for a condition and sets it"""
    expected_value = 'expected_%s' % condition
    self.resource_configuration[name]['required_attributes'].append(expected_value)
    self.resource_configuration[name]['conditions'][condition] = (self.resource_configuration[name]['conditions'][condition][0], self.parameters['attributes'].get(expected_value))