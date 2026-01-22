from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def UpdatePublicIPEnabled(unused_instance_ref, args, patch_request):
    """Hook to update public IP to the update mask of the request fo GA."""
    if args.IsSpecified('public_ip_enabled'):
        patch_request = AddFieldToUpdateMask('public_ip_enabled', patch_request)
    return patch_request