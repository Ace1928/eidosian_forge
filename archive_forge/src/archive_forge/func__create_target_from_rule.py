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
def _create_target_from_rule(client, rule, groups, vpc_id, tags, check_mode):
    owner_id = current_account_id
    if check_mode:
        return ('group', (owner_id, None, None), True)
    group_name = rule['group_name']
    try:
        created_group = _create_security_group_with_wait(client, group_name, rule['group_desc'], vpc_id, tags)
    except is_boto3_error_code('InvalidGroup.Duplicate'):
        fail_msg = f"Could not create or use existing group '{group_name}' in rule {rule}.  Make sure the group exists and try using the group_id instead of the name"
        return _lookup_target_or_fail(client, group_name, vpc_id, groups, fail_msg)
    except (BotoCoreError, ClientError) as e:
        raise SecurityGroupError(msg="Failed to create security group '{0}' in rule {1}", e=e)
    group_id = created_group['GroupId']
    groups[group_id] = created_group
    groups[group_name] = created_group
    return ('group', (owner_id, group_id, None), True)