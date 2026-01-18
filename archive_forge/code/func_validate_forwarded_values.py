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
def validate_forwarded_values(self, config, forwarded_values, cache_behavior):
    try:
        if not forwarded_values:
            forwarded_values = dict()
        existing_config = config.get('forwarded_values', {})
        headers = forwarded_values.get('headers', existing_config.get('headers', {}).get('items'))
        if headers:
            headers.sort()
        forwarded_values['headers'] = ansible_list_to_cloudfront_list(headers)
        if 'cookies' not in forwarded_values:
            forward = existing_config.get('cookies', {}).get('forward', self.__default_cache_behavior_forwarded_values_forward_cookies)
            forwarded_values['cookies'] = {'forward': forward}
        else:
            existing_whitelist = existing_config.get('cookies', {}).get('whitelisted_names', {}).get('items')
            whitelist = forwarded_values.get('cookies').get('whitelisted_names', existing_whitelist)
            if whitelist:
                self.validate_is_list(whitelist, 'forwarded_values.whitelisted_names')
                forwarded_values['cookies']['whitelisted_names'] = ansible_list_to_cloudfront_list(whitelist)
            cookie_forwarding = forwarded_values.get('cookies').get('forward', existing_config.get('cookies', {}).get('forward'))
            self.validate_attribute_with_allowed_values(cookie_forwarding, 'cache_behavior.forwarded_values.cookies.forward', self.__valid_cookie_forwarding)
            forwarded_values['cookies']['forward'] = cookie_forwarding
        query_string_cache_keys = forwarded_values.get('query_string_cache_keys', existing_config.get('query_string_cache_keys', {}).get('items', []))
        self.validate_is_list(query_string_cache_keys, 'forwarded_values.query_string_cache_keys')
        forwarded_values['query_string_cache_keys'] = ansible_list_to_cloudfront_list(query_string_cache_keys)
        forwarded_values = self.add_missing_key(forwarded_values, 'query_string', existing_config.get('query_string', self.__default_cache_behavior_forwarded_values_query_string))
        cache_behavior['forwarded_values'] = forwarded_values
        return cache_behavior
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating forwarded values')