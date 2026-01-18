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
def validate_origins(self, client, config, origins, default_origin_domain_name, default_origin_path, create_distribution, purge_origins=False):
    try:
        if origins is None:
            if default_origin_domain_name is None and (not create_distribution):
                if purge_origins:
                    return None
                else:
                    return ansible_list_to_cloudfront_list(config)
            if default_origin_domain_name is not None:
                origins = [{'domain_name': default_origin_domain_name, 'origin_path': default_origin_path or ''}]
            else:
                origins = []
        self.validate_is_list(origins, 'origins')
        if not origins and default_origin_domain_name is None and create_distribution:
            self.module.fail_json(msg='Both origins[] and default_origin_domain_name have not been specified. Please specify at least one.')
        all_origins = OrderedDict()
        new_domains = list()
        for origin in config:
            all_origins[origin.get('domain_name')] = origin
        for origin in origins:
            origin = self.validate_origin(client, all_origins.get(origin.get('domain_name'), {}), origin, default_origin_path)
            all_origins[origin['domain_name']] = origin
            new_domains.append(origin['domain_name'])
        if purge_origins:
            for domain in list(all_origins.keys()):
                if domain not in new_domains:
                    del all_origins[domain]
        return ansible_list_to_cloudfront_list(list(all_origins.values()))
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating distribution origins')