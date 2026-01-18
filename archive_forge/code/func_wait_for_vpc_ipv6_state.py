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
def wait_for_vpc_ipv6_state(module, connection, vpc_id, ipv6_assoc_state):
    """
    If ipv6_assoc_state is True, wait for VPC to be associated with at least one Amazon-provided IPv6 CIDR block.
    If ipv6_assoc_state is False, wait for VPC to be dissociated from all Amazon-provided IPv6 CIDR blocks.
    """
    if ipv6_assoc_state is None:
        return
    if module.check_mode:
        return
    start_time = time()
    criteria_match = False
    while time() < start_time + 300:
        current_value = get_vpc(module, connection, vpc_id)
        if current_value:
            ipv6_set = current_value.get('Ipv6CidrBlockAssociationSet')
            if ipv6_set:
                if ipv6_assoc_state:
                    for val in ipv6_set:
                        if val.get('Ipv6Pool') == 'Amazon' and val.get('Ipv6CidrBlockState').get('State') == 'associated':
                            criteria_match = True
                            break
                    if criteria_match:
                        break
                else:
                    expected_count = sum([val.get('Ipv6Pool') == 'Amazon' for val in ipv6_set])
                    actual_count = sum([val.get('Ipv6Pool') == 'Amazon' and val.get('Ipv6CidrBlockState').get('State') == 'disassociated' for val in ipv6_set])
                    if actual_count == expected_count:
                        criteria_match = True
                        break
        sleep(3)
    if not criteria_match:
        module.fail_json(msg='Failed to wait for IPv6 CIDR association')