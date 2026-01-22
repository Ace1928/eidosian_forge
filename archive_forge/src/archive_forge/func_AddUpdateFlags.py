from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def AddUpdateFlags(parser, support_location_hint=False, support_fleet=False, support_planning_status=False, support_local_ssd_count=False, support_share_setting=False, support_auto_delete=False, support_require_specific_reservation=False):
    """Adds all flags needed for the update command."""
    name_prefix_group = base.ArgumentGroup('Manage the name-prefix of a future reservation.', required=False, mutex=True)
    name_prefix_group.AddArgument(GetNamePrefixFlag())
    name_prefix_group.AddArgument(GetClearNamePrefixFlag())
    name_prefix_group.AddToParser(parser)
    GetTotalCountFlag(required=False).AddToParser(parser)
    reservation_flags.GetDescriptionFlag(is_fr=True).AddToParser(parser)
    if support_planning_status:
        GetPlanningStatusFlag().AddToParser(parser)
    group = base.ArgumentGroup('Manage the specific SKU reservation properties.', required=False)
    group.AddArgument(reservation_flags.GetMachineType(required=False))
    group.AddArgument(reservation_flags.GetMinCpuPlatform())
    accelerator_group = base.ArgumentGroup('Manage the accelerators of a future reservation.', required=False, mutex=True)
    accelerator_group.AddArgument(reservation_flags.GetAcceleratorFlag())
    accelerator_group.AddArgument(GetClearAcceleratorFlag())
    group.AddArgument(accelerator_group)
    local_ssd_group = base.ArgumentGroup('Manage the local ssd of a future reservation.', required=False, mutex=True)
    if support_local_ssd_count:
        local_ssd_group.AddArgument(reservation_flags.GetLocalSsdFlagWithCount())
    else:
        local_ssd_group.AddArgument(reservation_flags.GetLocalSsdFlag())
    local_ssd_group.AddArgument(GetClearLocalSsdFlag())
    group.AddArgument(local_ssd_group)
    if support_location_hint:
        group.AddArgument(reservation_flags.GetLocationHint())
    if support_fleet:
        group.AddArgument(instance_flags.AddMaintenanceInterval())
    group.AddToParser(parser)
    AddTimeWindowFlags(parser, time_window_requird=False)
    if support_share_setting:
        share_group = base.ArgumentGroup('Manage the properties of a shared future reservation.', required=False, mutex=True)
        share_group.AddArgument(GetClearShareSettingsFlag())
        share_setting_group = base.ArgumentGroup('Manage the share settings of a future reservation.', required=False)
        share_setting_group.AddArgument(GetSharedSettingFlag())
        share_setting_group.AddArgument(GetShareWithFlag())
        share_group.AddArgument(share_setting_group)
        share_group.AddToParser(parser)
    if support_auto_delete:
        AddAutoDeleteFlags(parser, is_update=True)
    if support_require_specific_reservation:
        GetRequireSpecificReservationFlag().AddToParser(parser)