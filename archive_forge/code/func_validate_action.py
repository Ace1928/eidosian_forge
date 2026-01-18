from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def validate_action(self, action):
    errors = []
    if action == 'create':
        if not self.parameters.get('role_name'):
            errors.append('role_name')
        if not self.parameters.get('applications'):
            errors.append('application_dicts or application_strs')
    if errors:
        plural = 's' if len(errors) > 1 else ''
        self.module.fail_json(msg='Error: missing required parameter%s for %s: %s.' % (plural, action, ' and: '.join(errors)))