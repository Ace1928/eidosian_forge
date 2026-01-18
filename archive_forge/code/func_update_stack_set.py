import datetime
import itertools
import time
import uuid
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_stack_set(module, stack_params, cfn):
    try:
        cfn.update_stack_set(**stack_params)
    except is_boto3_error_code('StackSetNotFound') as err:
        module.fail_json_aws(err, msg='Failed to find stack set. Check the name & region.')
    except is_boto3_error_code('StackInstanceNotFound') as err:
        module.fail_json_aws(err, msg='One or more stack instances were not found for this stack set. Double check the `accounts` and `regions` parameters.')
    except is_boto3_error_code('OperationInProgressException') as err:
        module.fail_json_aws(err, msg="Another operation is already in progress on this stack set - please try again later. When making multiple cloudformation_stack_set calls, it's best to enable `wait: true` to avoid unfinished op errors.")
    except (ClientError, BotoCoreError) as err:
        module.fail_json_aws(err, msg='Could not update stack set.')
    if module.params.get('wait'):
        await_stack_set_operation(module, cfn, operation_id=stack_params['OperationId'], stack_set_name=stack_params['StackSetName'], max_wait=module.params.get('wait_timeout'))
    return True