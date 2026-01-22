from __future__ import (absolute_import, division, print_function)
import traceback
import json
from ansible.errors import AnsibleError
from ansible.module_utils.urls import open_url
from ansible.plugins.inventory import (
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _stackpath_query_get_list(self, url):
    self._authenticate()
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.auth_token}
    next_page = True
    result = []
    cursor = '-1'
    while next_page:
        resp = open_url(url + '?page_request.first=10&page_request.after=%s' % cursor, headers=headers, method='GET')
        status_code = resp.code
        if status_code == 200:
            body = resp.read()
        body_json = json.loads(body)
        result.extend(body_json['results'])
        next_page = body_json['pageInfo']['hasNextPage']
        if next_page:
            cursor = body_json['pageInfo']['endCursor']
    return result