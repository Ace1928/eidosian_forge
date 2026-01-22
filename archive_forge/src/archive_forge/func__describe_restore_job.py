from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def _describe_restore_job(connection, module, restore_job_id):
    try:
        response = connection.describe_restore_job(RestoreJobId=restore_job_id)
        response.pop('ResponseMetadata', None)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Failed to describe restore job with ID: {restore_job_id}')
    return [camel_dict_to_snake_dict(response)]