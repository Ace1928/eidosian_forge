from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible_collections.community.dns.plugins.module_utils.argspec import (
from ansible_collections.community.dns.plugins.module_utils.json_api_helper import (
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def _extract_only_error_message(self, result):
    res = ''
    if isinstance(result.get('error'), dict):
        if 'message' in result['error']:
            res = '{0} with error message "{1}"'.format(res, result['error']['message'])
        if 'code' in result['error']:
            res = '{0} (error code {1})'.format(res, result['error']['code'])
    if result.get('message'):
        res = '{0} with message "{1}"'.format(res, result['message'])
    return res