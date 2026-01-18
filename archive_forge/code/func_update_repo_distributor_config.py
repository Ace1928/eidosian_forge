from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
def update_repo_distributor_config(self, repo_id, **kwargs):
    url = '%s/pulp/api/v2/repositories/%s/distributors/' % (self.host, repo_id)
    repo_config = self.get_repo_config_by_id(repo_id)
    for distributor in repo_config['distributors']:
        distributor_url = '%s%s/' % (url, distributor['id'])
        data = dict()
        data['distributor_config'] = dict()
        for key, value in kwargs.items():
            data['distributor_config'][key] = value
        response, info = fetch_url(self.module, distributor_url, data=json.dumps(data), method='PUT')
        if info['status'] != 202:
            self.module.fail_json(msg='Failed to set the relative url for the repository.', status_code=info['status'], response=info['msg'], url=url)