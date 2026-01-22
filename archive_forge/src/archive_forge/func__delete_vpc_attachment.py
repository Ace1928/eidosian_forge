from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
@Boto3Mixin.aws_error_handler('delete transit gateway attachment')
def _delete_vpc_attachment(self, **params):
    try:
        result = self.client.delete_transit_gateway_vpc_attachment(aws_retry=True, **params)
    except is_boto3_error_code('ResourceNotFoundException'):
        return None
    return result.get('TransitGatewayVpcAttachment', None)