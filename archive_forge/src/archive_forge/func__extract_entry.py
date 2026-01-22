from __future__ import absolute_import, division, print_function
import base64
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
def _extract_entry(dn, attrs, base64_attributes):
    extracted = {'dn': dn}
    for attr, val in list(attrs.items()):
        convert_to_base64 = '*' in base64_attributes or attr in base64_attributes
        if len(val) == 1:
            extracted[attr] = _normalize_string(val[0], convert_to_base64)
        else:
            extracted[attr] = [_normalize_string(v, convert_to_base64) for v in val]
    return extracted