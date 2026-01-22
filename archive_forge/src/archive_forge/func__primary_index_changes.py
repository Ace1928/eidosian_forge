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
def _primary_index_changes(current_table):
    primary_index = _decode_primary_index(current_table)
    hash_key_name = primary_index.get('hash_key_name')
    _hash_key_name = module.params.get('hash_key_name')
    hash_key_type = primary_index.get('hash_key_type')
    _hash_key_type = module.params.get('hash_key_type')
    range_key_name = primary_index.get('range_key_name')
    _range_key_name = module.params.get('range_key_name')
    range_key_type = primary_index.get('range_key_type')
    _range_key_type = module.params.get('range_key_type')
    changed = list()
    if _hash_key_name and _hash_key_name != hash_key_name:
        changed.append('hash_key_name')
    if _hash_key_type and _hash_key_type != hash_key_type:
        changed.append('hash_key_type')
    if _range_key_name and _range_key_name != range_key_name:
        changed.append('range_key_name')
    if _range_key_type and _range_key_type != range_key_type:
        changed.append('range_key_type')
    return changed