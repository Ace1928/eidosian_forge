from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
def _get_id_params(self, id=None, id_list=False):
    if not id:
        id = self.resource_id
    if not id:
        self.module.fail_json(msg='Attachment identifier parameter missing')
    if id_list:
        return dict(TransitGatewayAttachmentIds=[id])
    return dict(TransitGatewayAttachmentId=id)