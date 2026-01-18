from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_type_change(self, current):
    """present moving from onboard to external and reciprocally"""
    error = None
    if 'onboard' in current and 'external' in self.parameters:
        error = 'onboard key-manager is already installed, it needs to be deleted first.'
    if 'external' in current and 'onboard' in self.parameters:
        error = 'external key-manager is already installed, it needs to be deleted first.'
    if error:
        self.module.fail_json(msg='Error, cannot modify existing configuraton: %s' % error)