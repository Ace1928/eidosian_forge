from googlecloudsdk.command_lib.container.fleet import invalid_args_error
def RbacPolicyName(policy_name, project_id, membership, identity, is_user):
    """Generate RBAC policy name."""
    formatted_identity = FormatIdentityForResourceNaming(identity, is_user)
    if membership:
        metadata_name = project_id + '_' + formatted_identity + '_' + membership
    else:
        metadata_name = project_id + '_' + formatted_identity
    if policy_name == 'impersonate':
        return RBAC_IMPERSONATE_POLICY_NAME.format(metadata=metadata_name)
    if policy_name == 'permission':
        return RBAC_PERMISSION_POLICY_NAME.format(metadata=metadata_name)
    if policy_name == 'anthos':
        return RBAC_ANTHOS_SUPPORT_POLICY_NAME.format(metadata=metadata_name)