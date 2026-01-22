from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.util.args import labels_util
def AddCommonBulkInsertArgs(parser, release_track, deprecate_maintenance_policy=False, support_min_node_cpu=False, support_erase_vss=False, snapshot_csek=False, image_csek=False, support_display_device=False, support_local_ssd_size=False, support_numa_node_count=False, support_visible_core_count=False, support_max_run_duration=False, support_enable_target_shape=False, add_zone_region_flags=True, support_confidential_compute_type=False, support_confidential_compute_type_tdx=False, support_no_address_in_networking=False, support_max_count_per_zone=False, support_network_queue_count=False, support_performance_monitoring_unit=False, support_custom_hostnames=False, support_specific_then_x_affinity=False, support_ipv6_only=False, support_watchdog_timer=False, support_per_interface_stack_type=False):
    """Register parser args common to all tracks."""
    metadata_utils.AddMetadataArgs(parser)
    AddDiskArgsForBulk(parser)
    instances_flags.AddCreateDiskArgs(parser, enable_kms=True, enable_snapshots=True, source_snapshot_csek=snapshot_csek, image_csek=image_csek, include_name=False, support_boot=True)
    instances_flags.AddCanIpForwardArgs(parser)
    instances_flags.AddAcceleratorArgs(parser)
    instances_flags.AddMachineTypeArgs(parser)
    instances_flags.AddMaintenancePolicyArgs(parser, deprecate=deprecate_maintenance_policy)
    instances_flags.AddNoRestartOnFailureArgs(parser)
    instances_flags.AddPreemptibleVmArgs(parser)
    instances_flags.AddProvisioningModelVmArgs(parser)
    instances_flags.AddNetworkPerformanceConfigsArgs(parser)
    instances_flags.AddInstanceTerminationActionVmArgs(parser)
    instances_flags.AddServiceAccountAndScopeArgs(parser, False, extra_scopes_help='However, if neither `--scopes` nor `--no-scopes` are specified and the project has no default service account, then the instance will be created with no scopes. Note that the level of access that a service account has is determined by a combination of access scopes and IAM roles so you must configure both access scopes and IAM roles for the service account to work properly.')
    instances_flags.AddTagsArgs(parser)
    instances_flags.AddCustomMachineTypeArgs(parser)
    instances_flags.AddNoAddressArg(parser)
    instances_flags.AddNetworkArgs(parser)
    instances_flags.AddNetworkTierArgs(parser, instance=True)
    AddBulkCreateNetworkingArgs(parser, support_no_address_in_networking, support_network_queue_count=support_network_queue_count, support_per_interface_stack_type=support_per_interface_stack_type, support_ipv6_only=support_ipv6_only)
    instances_flags.AddImageArgs(parser, enable_snapshots=True)
    instances_flags.AddShieldedInstanceConfigArgs(parser)
    instances_flags.AddNestedVirtualizationArgs(parser)
    instances_flags.AddThreadsPerCoreArgs(parser)
    instances_flags.AddEnableUefiNetworkingArgs(parser)
    instances_flags.AddResourceManagerTagsArgs(parser)
    if support_numa_node_count:
        instances_flags.AddNumaNodeCountArgs(parser)
    if support_display_device:
        instances_flags.AddDisplayDeviceArg(parser)
    instances_flags.AddReservationAffinityGroup(parser, group_text='Specifies the reservation for the instance.', affinity_text='The type of reservation for the instance.', support_specific_then_x_affinity=support_specific_then_x_affinity)
    maintenance_flags.AddResourcePoliciesArgs(parser, 'added to', 'instance')
    if support_min_node_cpu:
        instances_flags.AddMinNodeCpuArg(parser)
    instances_flags.AddLocationHintArg(parser)
    if support_erase_vss:
        compute_flags.AddEraseVssSignature(parser, 'source snapshots or source machine image')
    labels_util.AddCreateLabelsFlags(parser)
    parser.add_argument('--description', help='Specifies a textual description of the instances.')
    base.ASYNC_FLAG.AddToParser(parser)
    parser.display_info.AddFormat('multi(instances:format="table(name,zone.basename())")')
    if support_visible_core_count:
        instances_flags.AddVisibleCoreCountArgs(parser)
    if support_local_ssd_size:
        instances_flags.AddLocalSsdArgsWithSize(parser)
    else:
        instances_flags.AddLocalSsdArgs(parser)
    if support_max_run_duration:
        instances_flags.AddMaxRunDurationVmArgs(parser)
        instances_flags.AddDiscardLocalSsdVmArgs(parser)
    if support_enable_target_shape:
        AddDistributionTargetShapeArgs(parser)
    instances_flags.AddStackTypeArgs(parser, support_ipv6_only)
    instances_flags.AddMinCpuPlatformArgs(parser, release_track)
    instances_flags.AddPublicDnsArgs(parser, instance=True)
    instances_flags.AddConfidentialComputeArgs(parser, support_confidential_compute_type, support_confidential_compute_type_tdx)
    instances_flags.AddPostKeyRevocationActionTypeArgs(parser)
    AddBulkCreateArgs(parser, add_zone_region_flags, support_max_count_per_zone, support_custom_hostnames)
    if support_performance_monitoring_unit:
        instances_flags.AddPerformanceMonitoringUnitArgs(parser)
    if support_watchdog_timer:
        instances_flags.AddWatchdogTimerArg(parser)