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
def _s3_args(s3_bucket, s3_key, s3_object_version):
    if not s3_bucket:
        return {}
    if not s3_key:
        return {}
    code = {'S3Bucket': s3_bucket, 'S3Key': s3_key}
    if s3_object_version:
        code.update({'S3ObjectVersion': s3_object_version})
    return code