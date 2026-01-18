import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def validate_invalidation_batch(self, invalidation_batch, caller_reference):
    try:
        if caller_reference is not None:
            valid_caller_reference = caller_reference
        else:
            valid_caller_reference = datetime.datetime.now().isoformat()
        valid_invalidation_batch = {'paths': self.create_aws_list(invalidation_batch), 'caller_reference': valid_caller_reference}
        return valid_invalidation_batch
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Error validating invalidation batch.')