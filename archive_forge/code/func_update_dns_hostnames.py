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
def update_dns_hostnames(connection, module, vpc_id, dns_hostnames):
    if dns_hostnames is None:
        return False
    current_dns_hostnames = connection.describe_vpc_attribute(Attribute='enableDnsHostnames', VpcId=vpc_id, aws_retry=True)['EnableDnsHostnames']['Value']
    if current_dns_hostnames == dns_hostnames:
        return False
    if module.check_mode:
        return True
    try:
        connection.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={'Value': dns_hostnames}, aws_retry=True)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, 'Failed to update enabled dns hostnames attribute')
    return True