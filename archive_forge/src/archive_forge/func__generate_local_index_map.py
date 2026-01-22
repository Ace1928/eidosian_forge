from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_indexes_active
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_table_exists
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_table_not_exists
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _generate_local_index_map(current_table):
    local_index_map = dict()
    existing_indexes = current_table['_local_index_map']
    for index in module.params.get('indexes'):
        if index.get('type') not in ['all', 'include', 'keys_only']:
            continue
        name = index.get('name')
        if name in local_index_map:
            module.fail_json(msg=f'Duplicate key {name} in list of local indexes')
        idx = _merge_index_params(index, existing_indexes.get(name, {}))
        idx['type'] = idx['type'].upper()
        local_index_map[name] = idx
    return local_index_map