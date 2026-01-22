from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def _update_organization_user_role(self, org_id, user_id, role):
    return self._api_call('PATCH', 'orgs/%d/users/%s' % (org_id, user_id), {'role': role})