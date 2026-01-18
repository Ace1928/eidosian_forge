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
def validate_cache_behavior(self, config, cache_behavior, valid_origins, is_default_cache=False):
    if is_default_cache and cache_behavior is None:
        cache_behavior = {}
    if cache_behavior is None and valid_origins is not None:
        return config
    cache_behavior = self.validate_cache_behavior_first_level_keys(config, cache_behavior, valid_origins, is_default_cache)
    if cache_behavior.get('cache_policy_id') is None:
        cache_behavior = self.validate_forwarded_values(config, cache_behavior.get('forwarded_values'), cache_behavior)
    cache_behavior = self.validate_allowed_methods(config, cache_behavior.get('allowed_methods'), cache_behavior)
    cache_behavior = self.validate_lambda_function_associations(config, cache_behavior.get('lambda_function_associations'), cache_behavior)
    cache_behavior = self.validate_trusted_signers(config, cache_behavior.get('trusted_signers'), cache_behavior)
    cache_behavior = self.validate_field_level_encryption_id(config, cache_behavior.get('field_level_encryption_id'), cache_behavior)
    return cache_behavior