from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def UpdatePscAllowedVpcs(unused_instance_ref, args, patch_request):
    """Hook to update psc confing allowed vpcs to the update mask of the request."""
    if args.IsSpecified('psc_allowed_vpcs'):
        _WarnForPscAllowedVpcsUpdate()
        patch_request.instance.pscConfig.allowedVpcs = args.psc_allowed_vpcs
        patch_request = AddFieldToUpdateMask('psc_config.allowed_vpcs', patch_request)
    elif args.IsSpecified('clear_psc_allowed_vpcs'):
        _WarnForPscAllowedVpcsRemovalUpdate()
        patch_request = AddFieldToUpdateMask('psc_config.allowed_vpcs', patch_request)
    return patch_request