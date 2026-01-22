from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def UpdateCustomDomain(unused_instance_ref, args, patch_request):
    """Hook to update custom domain to the update mask of the request."""
    if args.IsSpecified('custom_domain'):
        patch_request = AddFieldToUpdateMask('custom_domain', patch_request)
    return patch_request