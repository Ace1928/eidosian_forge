import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def update_key_rotation(connection, module, key, enable_key_rotation):
    if enable_key_rotation is None:
        return False
    key_id = key['key_arn']
    try:
        current_rotation_status = connection.get_key_rotation_status(KeyId=key_id)
        if current_rotation_status.get('KeyRotationEnabled') == enable_key_rotation:
            return False
    except is_boto3_error_code('AccessDeniedException'):
        pass
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Unable to get current key rotation status')
    if not module.check_mode:
        try:
            if enable_key_rotation:
                connection.enable_key_rotation(KeyId=key_id)
            else:
                connection.disable_key_rotation(KeyId=key_id)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to enable/disable key rotation')
    return True