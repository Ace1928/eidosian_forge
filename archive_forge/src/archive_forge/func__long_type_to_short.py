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
def _long_type_to_short(long_key):
    if not long_key:
        return None
    return DYNAMO_TYPE_MAP_LONG.get(long_key, None)