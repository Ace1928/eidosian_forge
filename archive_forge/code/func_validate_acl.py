from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def validate_acl(self):
    self.parameters = self.na_helper.filter_out_none_entries(self.parameters)
    if 'rights' in self.parameters:
        if 'advanced_rights' in self.parameters:
            self.module.fail_json(msg="Error: suboptions 'rights' and 'advanced_rights' are mutually exclusive.")
        self.module.warn('This module is not idempotent when "rights" is used, make sure to use "advanced_rights".')
    if not any((self.na_helper.safe_get(self.parameters, ['apply_to', key]) for key in self.apply_to_keys)):
        self.module.fail_json(msg='Error: at least one suboption must be true for apply_to.  Got: %s' % self.parameters.get('apply_to'))