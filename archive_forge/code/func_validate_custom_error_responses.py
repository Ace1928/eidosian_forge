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
def validate_custom_error_responses(self, config, custom_error_responses, purge_custom_error_responses):
    try:
        if custom_error_responses is None and (not purge_custom_error_responses):
            return ansible_list_to_cloudfront_list(config)
        self.validate_is_list(custom_error_responses, 'custom_error_responses')
        result = list()
        existing_responses = dict(((response['error_code'], response) for response in custom_error_responses))
        for custom_error_response in custom_error_responses:
            self.validate_required_key('error_code', 'custom_error_responses[].error_code', custom_error_response)
            custom_error_response = change_dict_key_name(custom_error_response, 'error_caching_min_ttl', 'error_caching_min_t_t_l')
            if 'response_code' in custom_error_response:
                custom_error_response['response_code'] = str(custom_error_response['response_code'])
            if custom_error_response['error_code'] in existing_responses:
                del existing_responses[custom_error_response['error_code']]
            result.append(custom_error_response)
        if not purge_custom_error_responses:
            result.extend(existing_responses.values())
        return ansible_list_to_cloudfront_list(result)
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating custom error responses')