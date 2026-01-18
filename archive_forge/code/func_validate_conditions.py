from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_conditions(self, name):
    conditions = self.resource_configuration[name].get('conditions')
    msgs = ['condition: %s is not valid for resource name: %s' % (condition, name) for condition in self.parameters['conditions'] if condition not in conditions]
    if msgs:
        msgs.append('valid condition%s: %s' % ('s are' if len(conditions) > 1 else ' is', ', '.join(conditions.keys())))
        self.module.fail_json(msg='Error: %s' % ', '.join(msgs))