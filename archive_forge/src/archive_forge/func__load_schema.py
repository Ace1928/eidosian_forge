from __future__ import absolute_import, division, print_function
import base64
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
def _load_schema(self):
    self.schema = self.module.params['schema']
    if self.schema:
        self.attrsonly = 1
    else:
        self.attrsonly = 0