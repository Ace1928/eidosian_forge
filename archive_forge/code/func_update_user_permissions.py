from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
def update_user_permissions(self, user_id, is_admin):
    url = '/api/admin/users/{user_id}/permissions'.format(user_id=user_id)
    permissions = dict(isGrafanaAdmin=is_admin)
    return self._send_request(url, data=permissions, headers=self.headers, method='PUT')