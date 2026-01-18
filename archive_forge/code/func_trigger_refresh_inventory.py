from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def trigger_refresh_inventory(rest_obj, slot_data):
    chassis_dict = dict([(slot['ChassisId'], slot['ChassisServiceTag']) for slot in slot_data.values()])
    jobs = []
    for chassis in chassis_dict:
        job_type = {'Id': 8, 'Name': 'Inventory_Task'}
        job_name = 'Refresh Inventory Chassis {0}'.format(chassis_dict[chassis])
        job_description = REFRESH_JOB_DESC
        target_param = [{'Id': int(chassis), 'Data': "''", 'TargetType': {'Id': 1000, 'Name': 'DEVICE'}}]
        job_params = [{'Key': 'operationName', 'Value': 'EC_SLOT_DEVICE_INVENTORY_REFRESH'}]
        job_resp = rest_obj.job_submission(job_name, job_description, target_param, job_params, job_type)
        job_id = job_resp.json_data.get('Id')
        jobs.append(int(job_id))
        time.sleep(SETTLING_TIME)
    return jobs