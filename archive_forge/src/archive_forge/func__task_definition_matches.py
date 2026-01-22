from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _task_definition_matches(requested_volumes, requested_containers, requested_task_role_arn, requested_launch_type, existing_task_definition):
    if td['status'] != 'ACTIVE':
        return None
    if requested_task_role_arn != td.get('taskRoleArn', ''):
        return None
    if requested_launch_type is not None and requested_launch_type not in td.get('requiresCompatibilities', []):
        return None
    existing_volumes = td.get('volumes', []) or []
    if len(requested_volumes) != len(existing_volumes):
        return None
    if len(requested_volumes) > 0:
        for requested_vol in requested_volumes:
            found = False
            for actual_vol in existing_volumes:
                if _right_has_values_of_left(requested_vol, actual_vol):
                    found = True
                    break
            if not found:
                return None
    existing_containers = td.get('containerDefinitions', []) or []
    if len(requested_containers) != len(existing_containers):
        return None
    for requested_container in requested_containers:
        found = False
        for actual_container in existing_containers:
            if _right_has_values_of_left(requested_container, actual_container):
                found = True
                break
        if not found:
            return None
    return existing_task_definition