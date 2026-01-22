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
def _throughput_changes(current_table, params=None):
    if not params:
        params = module.params
    throughput = current_table.get('provisioned_throughput', {})
    read_capacity = throughput.get('read_capacity_units', None)
    _read_capacity = params.get('read_capacity') or read_capacity
    write_capacity = throughput.get('write_capacity_units', None)
    _write_capacity = params.get('write_capacity') or write_capacity
    if read_capacity != _read_capacity or write_capacity != _write_capacity:
        return dict(ReadCapacityUnits=_read_capacity, WriteCapacityUnits=_write_capacity)
    return dict()