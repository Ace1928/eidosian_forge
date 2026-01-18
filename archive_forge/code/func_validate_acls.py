from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def validate_acls(self):
    if 'acls' not in self.parameters:
        return
    self.parameters['acls'] = self.na_helper.filter_out_none_entries(self.parameters['acls'])
    for acl in self.parameters['acls']:
        if 'rights' in acl:
            if 'advanced_rights' in acl:
                self.module.fail_json(msg="Error: suboptions 'rights' and 'advanced_rights' are mutually exclusive.")
            self.module.warn('This module is not idempotent when "rights" is used, make sure to use "advanced_rights".')
        if not any((self.na_helper.safe_get(acl, ['apply_to', key]) for key in self.apply_to_keys)):
            self.module.fail_json(msg='Error: at least one suboption must be true for apply_to.  Got: %s' % acl)
        self.match_acl_with_acls(acl, self.parameters['acls'])
    for option in ('access_control', 'ignore_paths', 'propagation_mode'):
        value = self.parameters.get(option)
        if value is not None:
            for acl in self.parameters['acls']:
                if acl.get(option) not in (None, value):
                    self.module.fail_json(msg='Error: mismatch between top level value and ACL value for %s: %s vs %s' % (option, value, acl.get(option)))
                acl[option] = value