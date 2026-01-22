from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.web_security_scanner import wss_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
def SetScanConfigAuth(unused_ref, args, request):
    """Modify request hook to set scan config details.

  Args:
    unused_ref: not used parameter to modify request hooks
    args: Parsed args namespace
    request: The partially filled request object.

  Returns:
    Request object with Authentication message filled out.
  """
    c = wss_base.WebSecurityScannerCommand()
    if args.auth_type is None:
        if any([args.auth_user, args.auth_password, args.auth_url]):
            raise exceptions.RequiredArgumentException('--auth-type', 'Required when setting authentication flags.')
        return request
    if args.auth_type == 'none':
        if any([args.auth_user, args.auth_password, args.auth_url]):
            raise exceptions.InvalidArgumentException('--auth-type', 'No other auth flags can be set with --auth-type=none')
        return request
    if request.scanConfig is None:
        request.scanConfig = c.messages.ScanConfig()
    request.scanConfig.authentication = c.messages.Authentication()
    if args.auth_type == 'google':
        _RequireAllFlagsOrRaiseForAuthType(args, ['auth_user', 'auth_password'], 'google')
        request.scanConfig.authentication.googleAccount = c.messages.GoogleAccount(username=args.auth_user, password=args.auth_password)
    elif args.auth_type == 'custom':
        _RequireAllFlagsOrRaiseForAuthType(args, ['auth_user', 'auth_password', 'auth_url'], 'custom')
        request.scanConfig.authentication.customAccount = c.messages.CustomAccount(username=args.auth_user, password=args.auth_password, loginUrl=args.auth_url)
    else:
        raise exceptions.UnknownArgumentException('--auth-type', args.auth_type)
    return request