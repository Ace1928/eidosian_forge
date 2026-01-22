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
def _generate_local_indexes():
    index_exists = dict()
    indexes = list()
    for index in module.params.get('indexes'):
        if index.get('type') not in ['all', 'include', 'keys_only']:
            continue
        name = index.get('name')
        if name in index_exists:
            module.fail_json(msg=f'Duplicate key {name} in list of local indexes')
        index['type'] = index['type'].upper()
        index = _generate_index(index, False)
        index_exists[name] = True
        indexes.append(index)
    return indexes