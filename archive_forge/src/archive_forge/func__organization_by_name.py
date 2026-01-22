from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def _organization_by_name(self, org_name):
    r, info = self._api_call('GET', 'orgs/name/%s' % org_name, None)
    if info['status'] != 200:
        raise GrafanaAPIException('Unable to retrieve organization: %s' % info)
    return json.loads(to_text(r.read()))