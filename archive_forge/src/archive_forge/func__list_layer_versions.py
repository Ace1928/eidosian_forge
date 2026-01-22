from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
@AWSRetry.jittered_backoff()
def _list_layer_versions(client, **params):
    paginator = client.get_paginator('list_layer_versions')
    return paginator.paginate(**params).build_full_result()