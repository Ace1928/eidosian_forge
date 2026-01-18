import json
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import add_role_to_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import create_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import delete_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_instance_profiles
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_role_attached_policies
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import remove_role_from_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def update_managed_policies(client, check_mode, role_name, managed_policies, purge_policies):
    if managed_policies is None:
        return False
    current_attached_policies = list_iam_role_attached_policies(client, role_name)
    current_attached_policies_arn_list = [policy['PolicyArn'] for policy in current_attached_policies]
    if len(managed_policies) == 1 and managed_policies[0] is None:
        managed_policies = []
    policies_to_remove = set(current_attached_policies_arn_list) - set(managed_policies)
    policies_to_remove = policies_to_remove if purge_policies else []
    policies_to_attach = set(managed_policies) - set(current_attached_policies_arn_list)
    changed = False
    if purge_policies and policies_to_remove:
        if check_mode:
            return True
        else:
            changed |= remove_policies(client, check_mode, policies_to_remove, role_name)
    if policies_to_attach:
        if check_mode:
            return True
        else:
            changed |= attach_policies(client, check_mode, policies_to_attach, role_name)
    return changed