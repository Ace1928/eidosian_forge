import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _create_website_configuration(suffix, error_key, redirect_all_requests):
    website_configuration = {}
    if error_key is not None:
        website_configuration['ErrorDocument'] = {'Key': error_key}
    if suffix is not None:
        website_configuration['IndexDocument'] = {'Suffix': suffix}
    if redirect_all_requests is not None:
        website_configuration['RedirectAllRequestsTo'] = _create_redirect_dict(redirect_all_requests)
    return website_configuration