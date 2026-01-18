from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def validate_taints(client, module, nodegroup, param_taints):
    changed = False
    params = dict()
    params['clusterName'] = nodegroup['clusterName']
    params['nodegroupName'] = nodegroup['nodegroupName']
    params['taints'] = []
    if 'taints' not in nodegroup:
        nodegroup['taints'] = []
    taints_to_add_or_update, taints_to_unset = compare_taints(nodegroup['taints'], param_taints)
    if taints_to_add_or_update:
        params['taints']['addOrUpdateTaints'] = taints_to_add_or_update
    if taints_to_unset:
        params['taints']['removeTaints'] = taints_to_unset
    if params['taints']:
        if not module.check_mode:
            changed = True
            try:
                client.update_nodegroup_config(**params)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg=f'Unable to set taints for Nodegroup {params['nodegroupName']}.')
    return changed