from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def AddCreateFlags(parser, support_location_hint=False, support_share_setting=False, support_fleet=False, support_instance_template=False, support_planning_status=False, support_local_ssd_count=False, support_auto_delete=False, support_require_specific_reservation=False):
    """Adds all flags needed for the create command."""
    GetNamePrefixFlag().AddToParser(parser)
    GetTotalCountFlag().AddToParser(parser)
    if support_require_specific_reservation:
        GetRequireSpecificReservationFlag().AddToParser(parser)
    reservation_flags.GetDescriptionFlag(is_fr=True).AddToParser(parser)
    if support_planning_status:
        GetPlanningStatusFlag().AddToParser(parser)
    specific_sku_properties_group = base.ArgumentGroup('Manage the instance properties for the auto-created reservations. You must either provide a source instance template or define the instance properties.', required=True, mutex=True)
    if support_instance_template:
        specific_sku_properties_group.AddArgument(reservation_flags.GetSourceInstanceTemplateFlag())
    AddTimeWindowFlags(parser, time_window_requird=True)
    instance_properties_group = base.ArgumentGroup('Define individual instance properties for the specific SKU reservation.')
    instance_properties_group.AddArgument(reservation_flags.GetMachineType())
    instance_properties_group.AddArgument(reservation_flags.GetMinCpuPlatform())
    if support_local_ssd_count:
        instance_properties_group.AddArgument(reservation_flags.GetLocalSsdFlagWithCount())
    else:
        instance_properties_group.AddArgument(reservation_flags.GetLocalSsdFlag())
    instance_properties_group.AddArgument(reservation_flags.GetAcceleratorFlag())
    if support_location_hint:
        instance_properties_group.AddArgument(reservation_flags.GetLocationHint())
    if support_fleet:
        instance_properties_group.AddArgument(instance_flags.AddMaintenanceFreezeDuration())
        instance_properties_group.AddArgument(instance_flags.AddMaintenanceInterval())
    specific_sku_properties_group.AddArgument(instance_properties_group)
    specific_sku_properties_group.AddToParser(parser)
    if support_share_setting:
        share_group = base.ArgumentGroup('Manage the properties of a shared reservation.', required=False)
        share_group.AddArgument(GetSharedSettingFlag())
        share_group.AddArgument(GetShareWithFlag())
        share_group.AddToParser(parser)
    if support_auto_delete:
        AddAutoDeleteFlags(parser)