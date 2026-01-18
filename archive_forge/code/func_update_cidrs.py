from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.network import to_subnet
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def update_cidrs(connection, module, vpc_obj, vpc_id, cidr_block, purge_cidrs):
    if cidr_block is None:
        return (False, None)
    associated_cidrs = dict(((cidr['CidrBlock'], cidr['AssociationId']) for cidr in vpc_obj.get('CidrBlockAssociationSet', []) if cidr['CidrBlockState']['State'] not in ['disassociating', 'disassociated']))
    current_cidrs = set(associated_cidrs.keys())
    desired_cidrs = set(cidr_block)
    if not purge_cidrs:
        desired_cidrs = desired_cidrs.union(current_cidrs)
    cidrs_to_add = list(desired_cidrs.difference(current_cidrs))
    cidrs_to_remove = list(current_cidrs.difference(desired_cidrs))
    if not cidrs_to_add and (not cidrs_to_remove):
        return (False, None)
    if module.check_mode:
        return (True, list(desired_cidrs))
    for cidr in cidrs_to_add:
        try:
            connection.associate_vpc_cidr_block(CidrBlock=cidr, VpcId=vpc_id, aws_retry=True)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, f'Unable to associate CIDR {cidr}.')
    for cidr in cidrs_to_remove:
        association_id = associated_cidrs[cidr]
        try:
            connection.disassociate_vpc_cidr_block(AssociationId=association_id, aws_retry=True)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, f'Unable to disassociate {association_id}. You must detach or delete all gateways and resources that are associated with the CIDR block before you can disassociate it.')
    return (True, list(desired_cidrs))