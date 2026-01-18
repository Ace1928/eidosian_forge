from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def verify_acls(connection, module, target_bucket):
    try:
        current_acl = connection.get_bucket_acl(aws_retry=True, Bucket=target_bucket)
        current_grants = current_acl['Grants']
    except is_boto3_error_code('NoSuchBucket'):
        module.fail_json(msg=f"Target Bucket '{target_bucket}' not found")
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to fetch target bucket ACL')
    required_grant = {'Grantee': {'URI': 'http://acs.amazonaws.com/groups/s3/LogDelivery', 'Type': 'Group'}, 'Permission': 'FULL_CONTROL'}
    for grant in current_grants:
        if grant == required_grant:
            return False
    if module.check_mode:
        return True
    updated_acl = dict(current_acl)
    updated_grants = list(current_grants)
    updated_grants.append(required_grant)
    updated_acl['Grants'] = updated_grants
    del updated_acl['ResponseMetadata']
    try:
        connection.put_bucket_acl(aws_retry=True, Bucket=target_bucket, AccessControlPolicy=updated_acl)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to update target bucket ACL to allow log delivery')
    return True