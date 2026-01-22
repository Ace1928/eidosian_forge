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
def _update_table(current_table):
    changes = dict()
    additional_global_index_changes = list()
    throughput_changes = _throughput_changes(current_table)
    if throughput_changes:
        changes['ProvisionedThroughput'] = throughput_changes
    current_billing_mode = current_table.get('billing_mode')
    new_billing_mode = module.params.get('billing_mode')
    if new_billing_mode is None:
        new_billing_mode = current_billing_mode
    if current_billing_mode != new_billing_mode:
        changes['BillingMode'] = new_billing_mode
    if module.params.get('table_class'):
        if module.params.get('table_class') != current_table.get('table_class'):
            changes['TableClass'] = module.params.get('table_class')
    global_index_changes = _global_index_changes(current_table)
    if global_index_changes:
        changes['GlobalSecondaryIndexUpdates'] = global_index_changes
        if current_billing_mode == new_billing_mode:
            if len(global_index_changes) > 1:
                changes['GlobalSecondaryIndexUpdates'] = [global_index_changes[0]]
                additional_global_index_changes = global_index_changes[1:]
    local_index_changes = _local_index_changes(current_table)
    if local_index_changes:
        changes['LocalSecondaryIndexUpdates'] = local_index_changes
    if not changes:
        return False
    if module.check_mode:
        return True
    if global_index_changes or local_index_changes:
        changes['AttributeDefinitions'] = _generate_attributes()
    try:
        client.update_table(aws_retry=True, TableName=module.params.get('name'), **changes)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to update table')
    if additional_global_index_changes:
        for index in additional_global_index_changes:
            wait_indexes()
            try:
                _update_table_with_long_retry(GlobalSecondaryIndexUpdates=[index], AttributeDefinitions=changes['AttributeDefinitions'])
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Failed to update table', changes=changes, additional_global_index_changes=additional_global_index_changes)
    return True