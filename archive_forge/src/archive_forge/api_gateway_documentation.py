import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
Make sure that we have the API configured and deployed as instructed.

    This function first configures the API correctly uploading the
    swagger definitions and then deploys those.  Configuration and
    deployment should be closely tied because there is only one set of
    definitions so if we stop, they may be updated by someone else and
    then we deploy the wrong configuration.
    