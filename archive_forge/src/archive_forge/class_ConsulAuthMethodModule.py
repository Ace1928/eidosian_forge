from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
class ConsulAuthMethodModule(_ConsulModule):
    api_endpoint = 'acl/auth-method'
    result_key = 'auth_method'
    unique_identifier = 'name'

    def map_param(self, k, v, is_update):
        if k == 'config' and v:
            v = {camel_case_key(k2): v2 for k2, v2 in v.items()}
        return super(ConsulAuthMethodModule, self).map_param(k, v, is_update)

    def needs_update(self, api_obj, module_obj):
        if 'MaxTokenTTL' in module_obj:
            module_obj['MaxTokenTTL'] = normalize_ttl(module_obj['MaxTokenTTL'])
        return super(ConsulAuthMethodModule, self).needs_update(api_obj, module_obj)