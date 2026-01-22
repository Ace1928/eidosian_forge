from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
class BaseTGWManager(BaseEc2Manager):

    @Boto3Mixin.aws_error_handler('connect to AWS')
    def _create_client(self, client_name='ec2'):
        if client_name == 'ec2':
            error_codes = ['IncorrectState']
        else:
            error_codes = []
        retry_decorator = AWSRetry.jittered_backoff(catch_extra_error_codes=error_codes)
        client = self.module.client(client_name, retry_decorator=retry_decorator)
        return client