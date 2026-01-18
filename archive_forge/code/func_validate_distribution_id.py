import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def validate_distribution_id(self, distribution_id, alias):
    try:
        if distribution_id is None and alias is None:
            self.module.fail_json(msg='distribution_id or alias must be specified')
        if distribution_id is None:
            distribution_id = self.__cloudfront_facts_mgr.get_distribution_id_from_domain_name(alias)
        return distribution_id
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Error validating parameters.')