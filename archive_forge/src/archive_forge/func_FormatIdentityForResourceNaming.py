from googlecloudsdk.command_lib.container.fleet import invalid_args_error
def FormatIdentityForResourceNaming(identity, is_user):
    """Format user by removing disallowed characters for k8s resource naming."""
    if is_user:
        desired_format = PRINCIPAL_USER_FORMAT
        error_message = invalid_args_error.INVALID_ARGS_USER_MESSAGE
    else:
        desired_format = PRINCIPAL_GROUP_FORMAT
        error_message = invalid_args_error.INVALID_ARGS_GROUP_MESSAGE
    parts = identity.split('/')
    if len(parts) >= 9:
        common_parts = parts[:4] + parts[5:8:2]
        if common_parts == desired_format:
            workforce_pool = identity.split('/workforcePools/')[1].split('/')[0]
            principal = identity.split('/{}/'.format(desired_format[-1]))[1]
            principal = principal.split('@')[0]
            resource_name = workforce_pool + '_' + principal
        else:
            raise invalid_args_error.InvalidArgsError(error_message)
    elif '@' not in identity:
        raise invalid_args_error.InvalidArgsError(error_message)
    else:
        resource_name = identity.split('@')[0]
    for ch in UNWANTED_CHARS:
        resource_name = resource_name.replace(ch, '')
    return resource_name