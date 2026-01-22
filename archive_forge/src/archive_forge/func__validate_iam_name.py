import re
from copy import deepcopy
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .arn import parse_aws_arn
from .arn import validate_aws_arn
from .botocore import is_boto3_error_code
from .botocore import normalize_boto3_result
from .errors import AWSErrorHandler
from .exceptions import AnsibleAWSError
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
def _validate_iam_name(resource_type, name=None):
    if name is None:
        return None
    LENGTHS = {'role': 64, 'user': 64}
    regex = '[\\w+=,.@-]+'
    max_length = LENGTHS.get(resource_type, 128)
    if len(name) > max_length:
        return f'Length of {resource_type} name may not exceed {max_length}'
    if not re.fullmatch(regex, name):
        return f'{resource_type} name must match pattern {regex}'
    return None