from __future__ import (absolute_import, division, print_function)
import json
import time
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def validate_delete_operation(rest_obj, module, catalog_list, delete_ids):
    associated_baselines = []
    for catalog in catalog_list:
        if catalog.get('AssociatedBaselines'):
            associated_baselines.append({'catalog_id': catalog['Id'], 'associated_baselines': catalog.get('AssociatedBaselines')})
        if catalog.get('Status') != 'Completed':
            resp = rest_obj.invoke_request('GET', JOB_URI.format(TaskId=catalog['TaskId']))
            job_data = resp.json_data
            if job_data['LastRunStatus']['Id'] == 2050:
                module.fail_json(msg=CATALOG_JOB_RUNNING.format(name=catalog['Name'], id=catalog['Id']), job_id=catalog['TaskId'])
    if associated_baselines:
        module.fail_json(msg=CATALOG_BASELINE_ATTACHED, associated_baselines=associated_baselines)
    if module.check_mode and len(catalog_list) > 0:
        module.exit_json(msg=CHECK_MODE_CHANGE_FOUND_MSG, changed=True, catalog_id=delete_ids)
    if len(catalog_list) == 0:
        module.exit_json(msg=CHECK_MODE_CHANGE_NOT_FOUND_MSG, changed=False)