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
def wait_for_lambda(client, module, name):
    try:
        client_active_waiter = client.get_waiter('function_active')
        client_updated_waiter = client.get_waiter('function_updated')
        client_active_waiter.wait(FunctionName=name)
        client_updated_waiter.wait(FunctionName=name)
    except WaiterError as e:
        module.fail_json_aws(e, msg='Timeout while waiting on lambda to finish updating')
    except (ClientError, BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed while waiting on lambda to finish updating')