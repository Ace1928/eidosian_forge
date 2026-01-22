import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _create_redirect_dict(url):
    redirect_dict = {}
    url_split = url.split(':')
    if len(url_split) == 2:
        redirect_dict['Protocol'] = url_split[0]
        redirect_dict['HostName'] = url_split[1].replace('//', '')
    elif len(url_split) == 1:
        redirect_dict['HostName'] = url_split[0]
    else:
        raise ValueError('Redirect URL appears invalid')
    return redirect_dict