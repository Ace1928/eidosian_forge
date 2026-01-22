from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
class TgwWaiterFactory(Ec2WaiterFactory):

    @property
    def _waiter_model_data(self):
        data = super(TgwWaiterFactory, self)._waiter_model_data
        tgw_data = dict(tgw_attachment_available=dict(operation='DescribeTransitGatewayAttachments', delay=5, maxAttempts=120, acceptors=[dict(state='success', matcher='pathAll', expected='available', argument='TransitGatewayAttachments[].State')]), tgw_attachment_deleted=dict(operation='DescribeTransitGatewayAttachments', delay=5, maxAttempts=120, acceptors=[dict(state='retry', matcher='pathAll', expected='deleting', argument='TransitGatewayAttachments[].State'), dict(state='success', matcher='pathAll', expected='deleted', argument='TransitGatewayAttachments[].State'), dict(state='success', matcher='path', expected=True, argument='length(TransitGatewayAttachments[]) == `0`'), dict(state='success', matcher='error', expected='InvalidRouteTableID.NotFound')]))
        data.update(tgw_data)
        return data