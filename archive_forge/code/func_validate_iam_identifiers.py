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
def validate_iam_identifiers(resource_type, name=None, path=None):
    name_problem = _validate_iam_name(resource_type, name)
    if name_problem:
        return name_problem
    path_problem = _validate_iam_path(resource_type, path)
    if path_problem:
        return path_problem
    return None