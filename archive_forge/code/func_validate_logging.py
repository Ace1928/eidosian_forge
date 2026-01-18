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
def validate_logging(self, logging):
    try:
        if logging is None:
            return None
        valid_logging = {}
        if logging and (not set(['enabled', 'include_cookies', 'bucket', 'prefix']).issubset(logging)):
            self.module.fail_json(msg='The logging parameters enabled, include_cookies, bucket and prefix must be specified.')
        valid_logging['include_cookies'] = logging.get('include_cookies')
        valid_logging['enabled'] = logging.get('enabled')
        valid_logging['bucket'] = logging.get('bucket')
        valid_logging['prefix'] = logging.get('prefix')
        return valid_logging
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating distribution logging')