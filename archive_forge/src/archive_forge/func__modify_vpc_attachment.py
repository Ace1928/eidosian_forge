from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
@Boto3Mixin.aws_error_handler('modify transit gateway attachment')
def _modify_vpc_attachment(self, **params):
    result = self.client.modify_transit_gateway_vpc_attachment(aws_retry=True, **params)
    return result.get('TransitGatewayVpcAttachment', None)