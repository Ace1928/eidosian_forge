from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
@AWSRetry.jittered_backoff(retries=10)
def _describe_alarms(connection, **params):
    paginator = connection.get_paginator('describe_alarms')
    return paginator.paginate(**params).build_full_result()