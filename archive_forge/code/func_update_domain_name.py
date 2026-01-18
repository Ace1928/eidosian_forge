import copy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(**retry_params)
def update_domain_name(client, domain_name, **kwargs):
    patch_operations = []
    for key, value in kwargs.items():
        path = '/' + key
        if key == 'endpointType':
            continue
        patch_operations.append({'op': 'replace', 'path': path, 'value': value})
    return client.update_domain_name(domainName=domain_name, patchOperations=patch_operations)