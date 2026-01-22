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
def ValidateBulkInsertArgs(args, support_enable_target_shape, support_source_snapshot_csek, support_image_csek, support_max_run_duration, support_max_count_per_zone, support_custom_hostnames):
    """Validates all bulk and instance args."""
    ValidateBulkCreateArgs(args)
    if support_enable_target_shape:
        ValidateBulkTargetShapeArgs(args)
    ValidateLocationPolicyArgs(args)
    if support_max_count_per_zone:
        ValidateMaxCountPerZoneArgs(args)
    if support_custom_hostnames:
        ValidateCustomHostnames(args)
    ValidateBulkDiskFlags(args, enable_source_snapshot_csek=support_source_snapshot_csek, enable_image_csek=support_image_csek)
    instances_flags.ValidateImageFlags(args)
    instances_flags.ValidateLocalSsdFlags(args)
    instances_flags.ValidateNicFlags(args)
    instances_flags.ValidateServiceAccountAndScopeArgs(args)
    instances_flags.ValidateAcceleratorArgs(args)
    instances_flags.ValidateNetworkTierArgs(args)
    instances_flags.ValidateReservationAffinityGroup(args)
    instances_flags.ValidateNetworkPerformanceConfigsArgs(args)
    instances_flags.ValidateInstanceScheduling(args, support_max_run_duration=support_max_run_duration)