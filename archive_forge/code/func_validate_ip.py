import itertools
import json
import re
from collections import namedtuple
from copy import deepcopy
from ipaddress import ip_network
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.network import to_ipv6_subnet
from ansible.module_utils.common.network import to_subnet
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_id
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def validate_ip(module, cidr_ip):
    split_addr = cidr_ip.split('/')
    if len(split_addr) != 2:
        return cidr_ip
    try:
        ip = ip_network(to_text(cidr_ip))
        return str(ip)
    except ValueError:
        pass
    try:
        ip = to_subnet(split_addr[0], split_addr[1])
        module.warn(f'One of your CIDR addresses ({cidr_ip}) has host bits set. To get rid of this warning, check the network mask and make sure that only network bits are set: {ip}.')
        return ip
    except ValueError:
        pass
    try:
        ip6 = to_ipv6_subnet(split_addr[0]) + '/' + split_addr[1]
        module.warn(f'One of your IPv6 CIDR addresses ({cidr_ip}) has host bits set. To get rid of this warning, check the network mask and make sure that only network bits are set: {ip6}.')
        return ip6
    except ValueError:
        module.warn(f'Unable to parse CIDR ({cidr_ip}).')
        return cidr_ip