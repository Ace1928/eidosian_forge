from operator import itemgetter
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
@AWSRetry.jittered_backoff(retries=MAX_AWS_RETRIES)
def _list_hosted_zones(route53, **kwargs):
    paginator = route53.get_paginator('list_hosted_zones')
    return paginator.paginate(**kwargs).build_full_result()['HostedZones']