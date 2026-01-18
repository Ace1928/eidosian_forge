import datetime
import re
from collections import OrderedDict
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def validate_distribution_from_caller_reference(self, caller_reference):
    try:
        distributions = self.__cloudfront_facts_mgr.list_distributions(keyed=False)
        distribution_name = 'Distribution'
        distribution_config_name = 'DistributionConfig'
        distribution_ids = [dist.get('Id') for dist in distributions]
        for distribution_id in distribution_ids:
            distribution = self.__cloudfront_facts_mgr.get_distribution(id=distribution_id)
            if distribution is not None:
                distribution_config = distribution[distribution_name].get(distribution_config_name)
                if distribution_config is not None and distribution_config.get('CallerReference') == caller_reference:
                    distribution[distribution_name][distribution_config_name] = distribution_config
                    return distribution
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating distribution from caller reference')