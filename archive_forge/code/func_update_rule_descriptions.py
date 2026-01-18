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
def update_rule_descriptions(module, client, group_id, present_ingress, named_tuple_ingress_list, present_egress, named_tuple_egress_list):
    changed = False
    ingress_needs_desc_update = []
    egress_needs_desc_update = []
    for present_rule in present_egress:
        needs_update = [r for r in named_tuple_egress_list if rule_cmp(r, present_rule) and r.description != present_rule.description]
        for r in needs_update:
            named_tuple_egress_list.remove(r)
        egress_needs_desc_update.extend(needs_update)
    for present_rule in present_ingress:
        needs_update = [r for r in named_tuple_ingress_list if rule_cmp(r, present_rule) and r.description != present_rule.description]
        for r in needs_update:
            named_tuple_ingress_list.remove(r)
        ingress_needs_desc_update.extend(needs_update)
    if ingress_needs_desc_update:
        update_rules_description(module, client, 'in', group_id, rules_to_permissions(ingress_needs_desc_update))
        changed |= True
    if egress_needs_desc_update:
        update_rules_description(module, client, 'out', group_id, rules_to_permissions(egress_needs_desc_update))
        changed |= True
    return changed