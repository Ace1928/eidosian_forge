import copy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_glue_job(connection, module, glue_job_name):
    """
    Get an AWS Glue job based on name. If not found, return None.

    :param connection: AWS boto3 glue connection
    :param module: Ansible module
    :param glue_job_name: Name of Glue job to get
    :return: boto3 Glue job dict or None if not found
    """
    try:
        return connection.get_job(aws_retry=True, JobName=glue_job_name)['Job']
    except is_boto3_error_code('EntityNotFoundException'):
        return None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e)