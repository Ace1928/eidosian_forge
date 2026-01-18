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
def validate_s3_origin_configuration(self, client, existing_config, origin):
    if origin.get('s3_origin_config', {}).get('origin_access_identity'):
        return origin['s3_origin_config']['origin_access_identity']
    if existing_config.get('s3_origin_config', {}).get('origin_access_identity'):
        return existing_config['s3_origin_config']['origin_access_identity']
    try:
        comment = f'access-identity-by-ansible-{origin.get('domain_name')}-{self.__default_datetime_string}'
        caller_reference = f'{origin.get('domain_name')}-{self.__default_datetime_string}'
        cfoai_config = dict(CloudFrontOriginAccessIdentityConfig=dict(CallerReference=caller_reference, Comment=comment))
        oai = client.create_cloud_front_origin_access_identity(**cfoai_config)['CloudFrontOriginAccessIdentity']['Id']
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg=f"Couldn't create Origin Access Identity for id {origin['id']}")
    return f'origin-access-identity/cloudfront/{oai}'