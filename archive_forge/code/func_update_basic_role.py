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
def update_basic_role(module, client, role_name, role):
    check_mode = module.check_mode
    assumed_policy = module.params.get('assume_role_policy_document')
    description = module.params.get('description')
    duration = module.params.get('max_session_duration')
    path = module.params.get('path')
    permissions_boundary = module.params.get('boundary')
    purge_tags = module.params.get('purge_tags')
    tags = module.params.get('tags')
    current_assumed_policy = role.get('AssumeRolePolicyDocument')
    current_description = role.get('Description')
    current_duration = role.get('MaxSessionDuration')
    current_permissions_boundary = role.get('PermissionsBoundary', {}).get('PermissionsBoundaryArn', '')
    current_tags = role.get('Tags', [])
    if update_role_path(client, check_mode, role, path):
        module.warn(f"iam_role doesn't support updating the path: current path '{role.get('Path')}', requested path '{path}'")
    changed = False
    changed |= update_role_tags(client, check_mode, role_name, tags, purge_tags, current_tags)
    changed |= update_role_assumed_policy(client, check_mode, role_name, assumed_policy, current_assumed_policy)
    changed |= update_role_description(client, check_mode, role_name, description, current_description)
    changed |= update_role_max_session_duration(client, check_mode, role_name, duration, current_duration)
    changed |= update_role_permissions_boundary(client, check_mode, role_name, permissions_boundary, current_permissions_boundary)
    return changed