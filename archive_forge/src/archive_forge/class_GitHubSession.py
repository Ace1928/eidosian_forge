from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
class GitHubSession(object):

    def __init__(self, module, token):
        self.module = module
        self.token = token

    def request(self, method, url, data=None):
        headers = {'Authorization': 'token %s' % self.token, 'Content-Type': 'application/json', 'Accept': 'application/vnd.github.v3+json'}
        response, info = fetch_url(self.module, url, method=method, data=data, headers=headers)
        if not 200 <= info['status'] < 400:
            self.module.fail_json(msg=' failed to send request %s to %s: %s' % (method, url, info['msg']))
        return GitHubResponse(response, info)