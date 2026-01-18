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
def validate_distribution_config_parameters(self, config, default_root_object, ipv6_enabled, http_version, web_acl_id):
    try:
        config['default_root_object'] = default_root_object or config.get('default_root_object', '')
        config['is_i_p_v6_enabled'] = ipv6_enabled if ipv6_enabled is not None else config.get('is_i_p_v6_enabled', self.__default_ipv6_enabled)
        if http_version is not None or config.get('http_version'):
            self.validate_attribute_with_allowed_values(http_version, 'http_version', self.__valid_http_versions)
            config['http_version'] = http_version or config.get('http_version')
        if web_acl_id or config.get('web_a_c_l_id'):
            config['web_a_c_l_id'] = web_acl_id or config.get('web_a_c_l_id')
        return config
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating distribution config parameters')