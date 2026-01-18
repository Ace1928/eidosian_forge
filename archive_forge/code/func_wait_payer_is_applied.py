import json
import time
from ansible.module_utils.basic import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.s3 import validate_bucket_name
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def wait_payer_is_applied(module, s3_client, bucket_name, expected_payer, should_fail=True):
    for dummy in range(0, 12):
        try:
            requester_pays_status = get_bucket_request_payment(s3_client, bucket_name)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed to get bucket request payment')
        if requester_pays_status != expected_payer:
            time.sleep(5)
        else:
            return requester_pays_status
    if should_fail:
        module.fail_json(msg='Bucket request payment failed to apply in the expected time', requested_status=expected_payer, live_status=requester_pays_status)
    else:
        return None