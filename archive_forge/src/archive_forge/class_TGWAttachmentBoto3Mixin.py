from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
class TGWAttachmentBoto3Mixin(Boto3Mixin):

    def __init__(self, module, **kwargs):
        self.tgw_waiter_factory = TgwWaiterFactory(module)
        super(TGWAttachmentBoto3Mixin, self).__init__(module, **kwargs)

    @AWSRetry.jittered_backoff()
    def _paginated_describe_transit_gateway_vpc_attachments(self, **params):
        paginator = self.client.get_paginator('describe_transit_gateway_vpc_attachments')
        return paginator.paginate(**params).build_full_result()

    @Boto3Mixin.aws_error_handler('describe transit gateway attachments')
    def _describe_vpc_attachments(self, **params):
        result = self._paginated_describe_transit_gateway_vpc_attachments(**params)
        return result.get('TransitGatewayVpcAttachments', None)

    @Boto3Mixin.aws_error_handler('create transit gateway attachment')
    def _create_vpc_attachment(self, **params):
        result = self.client.create_transit_gateway_vpc_attachment(aws_retry=True, **params)
        return result.get('TransitGatewayVpcAttachment', None)

    @Boto3Mixin.aws_error_handler('modify transit gateway attachment')
    def _modify_vpc_attachment(self, **params):
        result = self.client.modify_transit_gateway_vpc_attachment(aws_retry=True, **params)
        return result.get('TransitGatewayVpcAttachment', None)

    @Boto3Mixin.aws_error_handler('delete transit gateway attachment')
    def _delete_vpc_attachment(self, **params):
        try:
            result = self.client.delete_transit_gateway_vpc_attachment(aws_retry=True, **params)
        except is_boto3_error_code('ResourceNotFoundException'):
            return None
        return result.get('TransitGatewayVpcAttachment', None)

    @Boto3Mixin.aws_error_handler('transit gateway attachment to finish deleting')
    def _wait_tgw_attachment_deleted(self, **params):
        waiter = self.tgw_waiter_factory.get_waiter('tgw_attachment_deleted')
        waiter.wait(**params)

    @Boto3Mixin.aws_error_handler('transit gateway attachment to become available')
    def _wait_tgw_attachment_available(self, **params):
        waiter = self.tgw_waiter_factory.get_waiter('tgw_attachment_available')
        waiter.wait(**params)

    def _normalize_tgw_attachment(self, rtb):
        return self._normalize_boto3_resource(rtb)

    def _get_tgw_vpc_attachment(self, **params):
        attachments = self._describe_vpc_attachments(**params)
        if not attachments:
            return None
        attachment = attachments[0]
        return attachment