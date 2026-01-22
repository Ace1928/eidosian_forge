from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def access_tokens_equal(self):
    if self.access_token_object.name != self._module.params['name']:
        return False
    if self.access_token_object.scopes != self._module.params['scopes']:
        return False
    if self.access_token_object.access_level != ACCESS_LEVELS[self._module.params['access_level']]:
        return False
    if self.access_token_object.expires_at != self._module.params['expires_at']:
        return False
    return True