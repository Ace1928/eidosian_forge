from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes, to_text
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
import re
def _is_value_absent(self, name, value):
    """ True if the target attribute doesn't have the given value. """
    return not self._is_value_present(name, value)