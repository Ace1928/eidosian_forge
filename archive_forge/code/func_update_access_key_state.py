from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_access_keys
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_access_key
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
@IAMErrorHandler.common_error_handler('Failed to update access key for user')
def update_access_key_state(access_keys, user, access_key_id, enabled):
    keys = {k['access_key_id']: k for k in access_keys}
    if access_key_id not in keys:
        raise AnsibleIAMError(message=f'Access key "{access_key_id}" not found attached to User "{user}"')
    if enabled is None:
        return False
    access_key = keys.get(access_key_id)
    desired_status = 'Active' if enabled else 'Inactive'
    if access_key.get('status') == desired_status:
        return False
    if module.check_mode:
        return True
    client.update_access_key(aws_retry=True, UserName=user, AccessKeyId=access_key_id, Status=desired_status)
    return True