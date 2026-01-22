from typing import Union
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
def _list_backup_selections(client, module, plan_id):
    first_iteration = False
    next_token = None
    selections = []
    try:
        response = client.list_backup_selections(BackupPlanId=plan_id)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to list AWS backup selections')
    next_token = response.get('NextToken', None)
    if next_token is None:
        return response['BackupSelectionsList']
    while next_token:
        if first_iteration:
            try:
                response = client.list_backup_selections(BackupPlanId=plan_id, NextToken=next_token)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Failed to list AWS backup selections')
        first_iteration = True
        selections.append(response['BackupSelectionsList'])
        next_token = response.get('NextToken')