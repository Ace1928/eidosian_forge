import time
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import parse_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
def _subnets_to_vpc(self, subnets, subnet_details=None):
    if not subnets:
        return None
    if not subnet_details:
        subnet_details = self.ec2_manager._describe_subnets(SubnetIds=list(subnets))
    vpcs = [s.get('VpcId') for s in subnet_details]
    if len(set(vpcs)) > 1:
        self.module.fail_json(msg='Firewall subnets may only be in one VPC, multiple VPCs found', vpcs=list(set(vpcs)), subnets=subnet_details)
    return vpcs[0]