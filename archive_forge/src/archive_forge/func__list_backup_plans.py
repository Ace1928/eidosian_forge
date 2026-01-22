from typing import Union
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
def _list_backup_plans(client, backup_plan_name):
    first_iteration = False
    next_token = None
    response = client.list_backup_plans()
    next_token = response.get('NextToken', None)
    if next_token is None:
        entries = response['BackupPlansList']
        for backup_plan in entries:
            if backup_plan_name == backup_plan['BackupPlanName']:
                return backup_plan['BackupPlanId']
    while next_token is not None:
        if first_iteration:
            response = client.list_backup_plans(NextToken=next_token)
        first_iteration = True
        entries = response['BackupPlansList']
        for backup_plan in entries:
            if backup_plan_name == backup_plan['BackupPlanName']:
                return backup_plan['BackupPlanId']
        next_token = response.get('NextToken')