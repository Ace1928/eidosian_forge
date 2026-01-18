import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def update_grants(connection, module, key, desired_grants, purge_grants):
    existing_grants = key['grants']
    to_add, to_remove = compare_grants(existing_grants, desired_grants, purge_grants)
    if not (bool(to_add) or bool(to_remove)):
        return False
    key_id = key['key_arn']
    if not module.check_mode:
        for grant in to_remove:
            try:
                connection.retire_grant(KeyId=key_id, GrantId=grant['grant_id'])
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Unable to retire grant')
        for grant in to_add:
            grant_params = convert_grant_params(grant, key)
            try:
                connection.create_grant(**grant_params)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Unable to create grant')
    return True