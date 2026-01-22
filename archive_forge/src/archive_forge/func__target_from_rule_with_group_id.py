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
def _target_from_rule_with_group_id(rule, groups):
    owner_id = current_account_id
    FOREIGN_SECURITY_GROUP_REGEX = '^([^/]+)/?(sg-\\S+)?/(\\S+)'
    foreign_rule = re.match(FOREIGN_SECURITY_GROUP_REGEX, rule['group_id'])
    if not foreign_rule:
        return ('group', (owner_id, rule['group_id'], None), False)
    owner_id, group_id, group_name = foreign_rule.groups()
    group_instance = dict(UserId=owner_id, GroupId=group_id, GroupName=group_name)
    groups[group_id] = group_instance
    groups[group_name] = group_instance
    if group_id and group_name:
        if group_name.startswith('amazon-'):
            group_id = None
        else:
            group_name = None
    return ('group', (owner_id, group_id, group_name), False)