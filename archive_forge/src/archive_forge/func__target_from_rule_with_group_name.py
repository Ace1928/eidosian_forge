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
def _target_from_rule_with_group_name(client, rule, name, group, groups, vpc_id, tags, check_mode):
    group_name = rule['group_name']
    owner_id = current_account_id
    if group_name == name:
        group_id = group['GroupId']
        groups[group_id] = group
        groups[group_name] = group
        return ('group', (owner_id, group_id, None), False)
    if group_name in groups and group.get('VpcId') and groups[group_name].get('VpcId'):
        group_id = groups[group_name]['GroupId']
        return ('group', (owner_id, group_id, None), False)
    if group_name in groups and (not (group.get('VpcId') or groups[group_name].get('VpcId'))):
        group_id = groups[group_name]['GroupId']
        return ('group', (owner_id, group_id, None), False)
    if not rule.get('group_desc', '').strip():
        fail_msg = f"group '{group_name}' not found and would be automatically created by rule {rule} but no description was provided"
        return _lookup_target_or_fail(client, group_name, vpc_id, groups, fail_msg)
    return _create_target_from_rule(client, rule, groups, vpc_id, tags, check_mode)