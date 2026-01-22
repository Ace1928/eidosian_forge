import copy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _compare_glue_job_params(user_params, current_params):
    """
    Compare Glue job params. If there is a difference, return True immediately else return False

    :param user_params: the Glue job parameters passed by the user
    :param current_params: the Glue job parameters currently configured
    :return: True if any parameter is mismatched else False
    """
    if 'Description' not in current_params:
        current_params['Description'] = ''
    if 'DefaultArguments' not in current_params:
        current_params['DefaultArguments'] = dict()
    if 'AllocatedCapacity' in user_params and user_params['AllocatedCapacity'] != current_params['AllocatedCapacity']:
        return True
    if 'Command' in user_params:
        if user_params['Command']['ScriptLocation'] != current_params['Command']['ScriptLocation']:
            return True
        if user_params['Command']['PythonVersion'] != current_params['Command']['PythonVersion']:
            return True
    if 'Connections' in user_params and user_params['Connections'] != current_params['Connections']:
        return True
    if 'DefaultArguments' in user_params and user_params['DefaultArguments'] != current_params['DefaultArguments']:
        return True
    if 'Description' in user_params and user_params['Description'] != current_params['Description']:
        return True
    if 'ExecutionProperty' in user_params and user_params['ExecutionProperty']['MaxConcurrentRuns'] != current_params['ExecutionProperty']['MaxConcurrentRuns']:
        return True
    if 'GlueVersion' in user_params and user_params['GlueVersion'] != current_params['GlueVersion']:
        return True
    if 'MaxRetries' in user_params and user_params['MaxRetries'] != current_params['MaxRetries']:
        return True
    if 'Role' in user_params and user_params['Role'] != current_params['Role']:
        return True
    if 'Timeout' in user_params and user_params['Timeout'] != current_params['Timeout']:
        return True
    if 'GlueVersion' in user_params and user_params['GlueVersion'] != current_params['GlueVersion']:
        return True
    if 'WorkerType' in user_params and user_params['WorkerType'] != current_params['WorkerType']:
        return True
    if 'NumberOfWorkers' in user_params and user_params['NumberOfWorkers'] != current_params['NumberOfWorkers']:
        return True
    return False