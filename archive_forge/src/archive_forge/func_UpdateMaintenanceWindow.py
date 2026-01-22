from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def UpdateMaintenanceWindow(unused_instance_ref, args, patch_request):
    """Hook to update maintenance window to the update mask of the request."""
    if args.IsSpecified('maintenance_window_day') or args.IsSpecified('maintenance_window_time'):
        patch_request = AddFieldToUpdateMask('maintenance_window', patch_request)
    return patch_request