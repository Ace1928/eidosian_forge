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
def validate_cache_behaviors(self, config, cache_behaviors, valid_origins, purge_cache_behaviors=False):
    try:
        if cache_behaviors is None and valid_origins is not None and (purge_cache_behaviors is False):
            return ansible_list_to_cloudfront_list(config)
        all_cache_behaviors = OrderedDict()
        if not purge_cache_behaviors:
            for behavior in config:
                all_cache_behaviors[behavior['path_pattern']] = behavior
        for cache_behavior in cache_behaviors:
            valid_cache_behavior = self.validate_cache_behavior(all_cache_behaviors.get(cache_behavior.get('path_pattern'), {}), cache_behavior, valid_origins)
            all_cache_behaviors[cache_behavior['path_pattern']] = valid_cache_behavior
        if purge_cache_behaviors:
            for target_origin_id in set(all_cache_behaviors.keys()) - set([cb['path_pattern'] for cb in cache_behaviors]):
                del all_cache_behaviors[target_origin_id]
        return ansible_list_to_cloudfront_list(list(all_cache_behaviors.values()))
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating distribution cache behaviors')