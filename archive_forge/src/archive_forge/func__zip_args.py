import base64
import hashlib
import re
import traceback
from collections import Counter
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def _zip_args(zip_file, current_config, ignore_checksum):
    if not zip_file:
        return {}
    if not ignore_checksum:
        local_checksum = sha256sum(zip_file)
        remote_checksum = current_config.get('CodeSha256', '')
        if local_checksum == remote_checksum:
            return {}
    with open(zip_file, 'rb') as f:
        zip_content = f.read()
    return {'ZipFile': zip_content}