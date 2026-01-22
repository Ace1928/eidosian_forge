from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import is_boto3_error_code
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
def acm_catch_boto_exception(func):

    def runner(*args, **kwargs):
        module = kwargs.pop('module', None)
        error = kwargs.pop('error', None)
        ignore_error_codes = kwargs.pop('ignore_error_codes', [])
        try:
            return func(*args, **kwargs)
        except is_boto3_error_code(ignore_error_codes):
            return None
        except (BotoCoreError, ClientError) as e:
            if not module:
                raise
            module.fail_json_aws(e, msg=error)
    return runner