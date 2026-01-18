from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict, job_tracking
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import CHANGES_MSG, NO_CHANGES_MSG
def update_common_job(module, payload, task, valid_ids):
    payload['Schedule'] = module.params.get('job_schedule')
    if module.params.get('job_name'):
        payload['JobName'] = module.params.get('job_name')
    else:
        payload['JobName'] = jobname_map.get(task)
    if module.params.get('job_description'):
        payload['JobDescription'] = module.params.get('job_description')
    else:
        payload['JobDescription'] = JOB_DESC.format(jobname_map.get(task), ','.join(map(str, valid_ids)))