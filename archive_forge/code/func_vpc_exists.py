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
def vpc_exists(module, vpc, name, cidr_block, multi):
    """Returns None or a vpc object depending on the existence of a VPC. When supplied
    with a CIDR, it will check for matching tags to determine if it is a match
    otherwise it will assume the VPC does not exist and thus return None.
    """
    try:
        vpc_filters = ansible_dict_to_boto3_filter_list({'tag:Name': name, 'cidr-block': cidr_block})
        matching_vpcs = vpc.describe_vpcs(aws_retry=True, Filters=vpc_filters)['Vpcs']
        if not matching_vpcs:
            vpc_filters = ansible_dict_to_boto3_filter_list({'tag:Name': name, 'cidr-block': [cidr_block[0]]})
            matching_vpcs = vpc.describe_vpcs(aws_retry=True, Filters=vpc_filters)['Vpcs']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe VPCs')
    if multi:
        return None
    elif len(matching_vpcs) == 1:
        return matching_vpcs[0]['VpcId']
    elif len(matching_vpcs) > 1:
        module.fail_json(msg=f'Currently there are {len(matching_vpcs)} VPCs that have the same name and CIDR block you specified. If you would like to create the VPC anyway please pass True to the multi_ok param.')
    return None