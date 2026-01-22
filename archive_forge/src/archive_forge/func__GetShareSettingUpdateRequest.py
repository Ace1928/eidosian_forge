from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.reservations import flags as r_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.compute.reservations import util
def _GetShareSettingUpdateRequest(args, reservation_ref, holder, support_share_with_flag):
    """Create Update Request for share-with.

  Returns:
  update request.
  Args:
   args: The arguments given to the update command.
   reservation_ref: reservation refrence.
   holder: base_classes.ComputeApiHolder.
   support_share_with_flag: Check if share_with is supported.
  """
    messages = holder.client.messages
    share_settings = None
    setting_configs = 'projects'
    if support_share_with_flag:
        if args.IsSpecified('share_with'):
            share_settings = util.MakeShareSettingsWithArgs(messages, args, setting_configs, share_with='share_with')
            update_mask = ['shareSettings.projectMap.' + project for project in getattr(args, 'share_with', [])]
    if args.IsSpecified('add_share_with'):
        share_settings = util.MakeShareSettingsWithArgs(messages, args, setting_configs, share_with='add_share_with')
        update_mask = ['shareSettings.projectMap.' + project for project in getattr(args, 'add_share_with', [])]
    elif args.IsSpecified('remove_share_with'):
        share_settings = messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.SPECIFIC_PROJECTS)
        update_mask = ['shareSettings.projectMap.' + project for project in getattr(args, 'remove_share_with', [])]
    r_resource = util.MakeReservationMessage(messages, reservation_ref.Name(), share_settings, None, None, None, reservation_ref.zone)
    r_update_request = messages.ComputeReservationsUpdateRequest(reservation=reservation_ref.Name(), reservationResource=r_resource, paths=update_mask, project=reservation_ref.project, zone=reservation_ref.zone)
    return r_update_request